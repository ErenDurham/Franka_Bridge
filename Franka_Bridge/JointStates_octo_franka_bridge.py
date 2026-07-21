"""
Written by Eren D.
Code to bridge the octo model and the Franka robot
Utilizing sensor_msgs/JointState published to /gello/joint_states to command joint deltas

To run:
conda activate octo
python3 src/Franka_Bridge/JointStates_octo_franka_bridge.py --checkpoint_weights_path="/home/faro/octo_fr3/checkpoints_fixed/octo_fr3_finetune/experiment_20260716_110912" --checkpoint_step="50000"
    For models not finetuned on deltas: add --action_mode=abs
"""

import threading
import time
import numpy as np
from functools import partial
from octo.utils.train_callbacks import supply_rng
import jax
import jax.numpy as jnp
import cv2
import rclpy
import rclpy.executors
from absl import app, flags as absl_flags
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from rclpy.action import ActionClient
from franka_msgs.action import Move as GripperMove
from octo.model.octo_model import OctoModel
import click

flags = absl_flags.FLAGS

topic_image_primary = "/camera/primary/image_raw"
topic_image_wrist = "/camera/wrist/image_raw"
topic_joint_states = "/joint_states"  # robot's actual state
topic_joint_cmd = "/gello/joint_states"  # desired joint positions
topic_gripper_state = "/franka_gripper/joint_states"  # actual finger positions

img_size_primary = 256
img_size_wrist = 128
window_size = 1

# 0.0 = fully closed and 1.0 = fully open.
default_gripper_command_topic = "/gripper/gripper_client/target_gripper_width_percent"

step_duration = 0.1  # (10 Hz) step size
max_timesteps = 500

# Safety clamp for delta mode
max_joint_delta = 0.2

# Gripper controls
gripper_open_width = 0.08  # meters
gripper_close_width = 0.0
gripper_speed = 0.1  # m/s
gripper_force = 10.0  # N
gripper_threshold = 0.5
gripper_latch_count = 2
gripper_hold_time = 1.5

joint_limits = [
    [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895],
    [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895],
]

demo_joint_min = np.array([-0.25, -0.272, -0.646, -2.894, -0.237, 1.101, -3.026])
demo_joint_max = np.array([0.176, 1.011, 0.489, -1.103, 0.984, 3.125, 1.277])
ood_margin = 0.3

fr3_joints = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

home_joints = np.array([-0.1308, 0.2754, 0.01304, -1.5552, 0.0575, 1.8988, -1.0340])

absl_flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
absl_flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)
absl_flags.DEFINE_enum(
    "action_mode",
    "delta",
    ["delta", "abs"],
    "delta: actions are joint deltas (fr3_dataset_transform_delta checkpoints); "
    "abs: actions are absolute joint targets (older checkpoints)",
)


class OctoFrankaBridge(Node):
    def __init__(self):
        super().__init__("octo_franka_bridge")

        # image and state buffers
        self._image_primary = None
        self._image_wrist = None
        self._current_joints = None
        self._gripper_width = 0.0
        self._gripper_is_open = None
        self._gripper_pending_count = 0
        self._gripper_last_change = 0.0
        self.model = None

        # Subscribers

        # subscribe to primary and wrist camera images
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, topic_image_primary, self._cb_image_primary, image_qos)
        self.create_subscription(Image, topic_image_wrist, self._cb_image_wrist, image_qos)

        # get joint positions
        self.create_subscription(
            JointState, topic_joint_states, self._cb_joint_states, 10
        )

        # get gripper finger positions (width = sum of both fingers)
        self.create_subscription(
            JointState, topic_gripper_state, self._cb_gripper_state, 10
        )

        # Publishers
        self._joint_pub = self.create_publisher(JointState, topic_joint_cmd, 10)
        self._grip_move_cli = ActionClient(self, GripperMove, "/franka_gripper/move")

        self._joints_desired = home_joints.copy()
        self._joints_lock = threading.Lock()
        self._ramp_start = None
        self._ramp_target = None
        self._ramp_t0 = 0.0
        self._ramp_duration = 3.0

    def _cb_image_primary(self, msg: Image) -> None:
        # store latest primary image
        self._image_primary = msg

    def _cb_image_wrist(self, msg: Image) -> None:
        # store latest wrist image
        self._image_wrist = msg

    def _cb_joint_states(self, msg: JointState) -> None:
        # extract arm joints by name to handle arbitrary ordering in the message
        if not msg.name:
            return
        try:
            idx = [msg.name.index(j) for j in fr3_joints]
            self._current_joints = np.array(msg.position)[idx]
        except ValueError:
            self._current_joints = np.array(msg.position[:7])

    def _cb_gripper_state(self, msg: JointState) -> None:
        # franka_gripper publishes width/2 on each of the two finger joints
        if len(msg.position) >= 2:
            self._gripper_width = float(msg.position[0] + msg.position[1])
        elif len(msg.position) == 1:
            self._gripper_width = float(msg.position[0])

    def _timer_publish_joints(self) -> None:
        with self._joints_lock:
            if self._ramp_target is not None:
                alpha = min(1.0, (time.time() - self._ramp_t0) / self._ramp_duration)
                self._joints_desired = (
                    self._ramp_start + alpha * (self._ramp_target - self._ramp_start)
                )
                if alpha >= 1.0:
                    self._ramp_target = None
            joints = self._joints_desired.copy()
        self.publish_joint_state(joints)

    def set_desired_joints(self, joints: np.ndarray) -> None:
        with self._joints_lock:
            self._ramp_target = None  # cancel any active ramp
            self._joints_desired = joints.copy()

    def set_ramp_target(self, target: np.ndarray, duration: float = 3.0) -> None:
        """
        Start a non-blocking ramp toward desired pose
        """
        with self._joints_lock:
            self._ramp_start = self._joints_desired.copy()
            self._ramp_target = np.asarray(target, dtype=np.float64).copy()
            self._ramp_t0 = time.time()
            self._ramp_duration = max(float(duration), 1e-3)

    def load_model(self) -> None:
        self.model = OctoModel.load_pretrained(
            flags.checkpoint_weights_path,
            flags.checkpoint_step,
        )
        keys = list(self.model.dataset_statistics.keys())
        self.get_logger().info(f"Dataset statistics keys: {keys}")
        if "action" not in self.model.dataset_statistics:
            raise KeyError(f"'action' not in dataset_statistics. Available: {keys}")

        # gets the stats
        if "proprio" not in self.model.dataset_statistics:
            raise KeyError(f"'proprio' not in dataset_statistics. Available: {keys}")
        proprio_stats = self.model.dataset_statistics["proprio"]
        self._proprio_mean = np.array(proprio_stats["mean"], dtype=np.float32)
        self._proprio_std = np.array(proprio_stats["std"], dtype=np.float32)

        action_mean = np.abs(
            np.array(self.model.dataset_statistics["action"]["mean"])[:7]
        )
        looks_delta = action_mean.max() < 0.05
        if flags.action_mode == "delta" and not looks_delta:
            raise ValueError(
                "action_mode=delta but checkpoint action means look ABSOLUTE "
                f"({action_mean.round(3)}); rerun with --action_mode=abs"
            )
        if flags.action_mode == "abs" and looks_delta:
            raise ValueError(
                "action_mode=abs but checkpoint action means look like DELTAS "
                f"({action_mean.round(3)}); rerun with --action_mode=delta"
            )

    def start_controller(self) -> None:
        # spin until first joint state message arrives
        while self._current_joints is None:
            time.sleep(0.1)
            self.get_logger().info("Waiting for joint states...")
        self.create_timer(0.1, self._timer_publish_joints)
        self.get_logger().info("Joint states received, controller ready")

    def stop_controller(self) -> None:
        self.get_logger().info("Episode stopped, holding last joint position.")

    def toggle_servo(self, start=True):
        if start:
            if self._current_joints is not None:
                self.ramp_to(self._current_joints, duration=step_duration)
            self.get_logger().info("Joint controller started")
        else:
            self.get_logger().info("Joint controller stopped, holding position")

    def ramp_to(self, target: np.ndarray, duration: float = 3.0) -> None:
        """
        Moves to target pose over the desired seconds
        """
        self.set_ramp_target(target, duration)
        time.sleep(duration)

    def go_home(self) -> None:
        """
        Ramp to home_joints over 3s.
        """
        self.ramp_to(home_joints, duration=3.0)
        self.get_logger().info("go_home complete")

    def _ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        """
        Convert a ROS Image message to a (H, W, 3) uint8 RGB numpy array
        """
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        arr = arr[:, :, :3]
        if msg.encoding.lower().startswith("bgr"):
            arr = arr[:, :, ::-1]

        # cv2.imshow("Camera", arr)
        # cv2.waitKey(1)
        return np.ascontiguousarray(arr)

    def open_gripper(self) -> None:
        self.send_gripper(1.0)

    def close_gripper(self) -> None:
        self.send_gripper(0.0)

    def publish_joint_state(self, joints_desired: np.ndarray) -> None:
        """Publish desired joint positions as JointState to /gello/joint_states."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = fr3_joints
        msg.position = joints_desired.tolist()
        msg.velocity = []
        msg.effort = []
        self._joint_pub.publish(msg)

    def send_gripper(self, gripper_value: float) -> None:
        """
        CLI cmd:
            ros2 action send_goal /franka_gripper/move franka_msgs/action/Move "{width: 0.08, speed: 0.1}"
            ros2 action send_goal /franka_gripper/move franka_msgs/action/Move "{width: 0.0, speed: 0.1}"
        """
        target_open = gripper_value > gripper_threshold
        if target_open == self._gripper_is_open:
            self._gripper_pending_count = 0
            return

        if self._gripper_is_open is not None:
            self._gripper_pending_count += 1
            if self._gripper_pending_count < gripper_latch_count:
                return
            if time.time() - self._gripper_last_change < gripper_hold_time:
                return

        # find if topic is available
        if not self._grip_move_cli.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Gripper not available!! :(")
            return

        goal = GripperMove.Goal()
        goal.width = gripper_open_width if target_open else gripper_close_width
        goal.speed = gripper_speed

        future = self._grip_move_cli.send_goal_async(goal)
        self._gripper_is_open = target_open
        self._gripper_pending_count = 0
        self._gripper_last_change = time.time()

        self.get_logger().info(f"send_gripper: {'open' if target_open else 'close'}")


    def read_state(self) -> dict:
        q = self._current_joints if self._current_joints is not None else np.zeros(7)

        # checkpoint expects 8-dim proprio: 7 arm joints + 1 gripper width
        proprio = np.concatenate([q, [self._gripper_width]]).astype(np.float32)

        # normalize to match training
        proprio = (proprio - self._proprio_mean) / (self._proprio_std + 1e-8)
        return {
            "joints": q,
            "proprio": proprio,
        }

    def safety_check(self) -> bool:
        if self._current_joints is None:
            return True
        lo, hi = joint_limits
        return all(lo[i] <= self._current_joints[i] <= hi[i] for i in range(7))

    def joints_in_envelope(self) -> bool:
        if self._current_joints is None:
            return True
        lo = demo_joint_min - ood_margin
        hi = demo_joint_max + ood_margin
        return bool(np.all(self._current_joints >= lo) and np.all(self._current_joints <= hi))

    def grab_image(self):
        # return (primary, wrist), substituting null images per-camera if unavailable
        if self._image_primary is None:
            self.get_logger().info("Primary image not found. Using null image....")
            primary = np.zeros((img_size_primary, img_size_primary, 3), dtype=np.uint8)
        else:
            primary = self._image_primary
        if self._image_wrist is None:
            self.get_logger().info("Wrist image not found. Using null image....")
            wrist = np.zeros((img_size_wrist, img_size_wrist, 3), dtype=np.uint8)
        else:
            wrist = self._image_wrist
        return primary, wrist

    def chunk_to_targets(self, action_chunk: np.ndarray) -> np.ndarray:
        """
        Convert a model action chunk (action_horizon, 8) into absolute joint
        targets (action_horizon, 7), clipped to joint_limits.
        """
        lo = np.array(joint_limits[0])
        hi = np.array(joint_limits[1])

        if flags.action_mode == "delta":
            deltas = np.clip(action_chunk[:, :7], -max_joint_delta, max_joint_delta)
            targets = self._current_joints[None, :] + np.cumsum(deltas, axis=0)
        else:
            targets = action_chunk[:, :7]

        return np.clip(targets, lo, hi)

    # Main loop
    def main(self, window_size: int = window_size) -> None:
        """
        Runs inference episode
        Builds function, obtains task/goal image, and runs the inference
        """

        # Build the inference function
        def _sample_actions(pretrained_model, observations, tasks, rng):
            observations = jax.tree_map(lambda x: x[None], observations)
            actions = pretrained_model.sample_actions(
                observations,
                tasks,
                rng=rng,
                unnormalization_statistics=pretrained_model.dataset_statistics["action"],
            )
            # sample_actions returns (batch, action_horizon, action_dim);
            # [0] selects the first batch item -> full chunk (action_horizon, action_dim)
            return actions[0]

        policy_fn = supply_rng(partial(_sample_actions, self.model))

        goal_image = jnp.zeros((img_size_primary, img_size_primary, 3), dtype=np.uint8)
        goal_instruction = "Pick up the green pepper and place it in bin"

        while True:
            self.go_home()
            # Goal selection
            modality = click.prompt(
                "Language or goal image?", type=click.Choice(["l", "g"])
            )

            # Goal image chosen
            if modality == "g":
                # Check if new goal image is needed
                if click.confirm("Take a new goal?", default=True):
                    input("Move arm to goal pose, then press [Enter] to capture.")
                    primary, _ = self.grab_image()
                    if isinstance(primary, Image):
                        raw = self._ros_image_to_numpy(primary)  # format image
                        goal_image = cv2.resize(raw, (img_size_primary, img_size_primary))
                    else:
                        goal_image = primary  # null fallback from grab_image
                task = self.model.create_tasks(
                    goals={"image_primary": goal_image[None]}
                )
                goal_instruction = ""  # empty goal instr

            # Language instruction chosen
            elif modality == "l":
                self.get_logger().info(f"Current instruction: {goal_instruction}")
                # Check if new language instr is desired
                if click.confirm("Enter a new instruction?", default=True):
                    goal_instruction = input("Instruction? ")
                task = self.model.create_tasks(texts=[goal_instruction])
                goal_image = jnp.zeros(
                    (img_size_primary, img_size_primary, 3), dtype=np.uint8
                )  # empty goal image

            else:
                raise NotImplementedError()

            input("Press [Enter] to start.")
            self.toggle_servo(start=True)

            ########## Control loop  ###############

            obs_history: list[dict] = []
            last_tstep = time.time()
            truncated = False
            step_count = 0

            while step_count < max_timesteps:
                # build observations from latest camera frames
                prev_stamp = getattr(self, "_last_img_stamp", None)

                primary, wrist = self.grab_image()
                img_primary = (
                    cv2.resize(self._ros_image_to_numpy(primary), (img_size_primary, img_size_primary))
                    if isinstance(primary, Image)
                    else primary
                )
                img_wrist = (
                    cv2.resize(self._ros_image_to_numpy(wrist), (img_size_wrist, img_size_wrist))
                    if isinstance(wrist, Image)
                    else wrist
                )

                # add to the observation history
                state = self.read_state()
                obs_history.append({
                    "image_primary": img_primary,
                    "image_wrist": img_wrist,
                    "proprio": state["proprio"],
                })
                if len(obs_history) > window_size:
                    obs_history.pop(0)

                # pad to window_size by repeating the earliest frame
                pad_count = window_size - len(obs_history)
                pad = obs_history[0:1] * pad_count
                windowed = pad + obs_history
                obs = {
                    "image_primary": np.stack([o["image_primary"] for o in windowed]),
                    "image_wrist": np.stack([o["image_wrist"] for o in windowed]),
                    "proprio": np.stack([o["proprio"] for o in windowed]),
                    "timestep_pad_mask": np.array(
                        [False] * pad_count + [True] * len(obs_history), dtype=bool
                    ),
                }

                # model inference — full action chunk (action_horizon, 8)
                t0 = time.time()
                action_chunk = np.array(policy_fn(obs, task), dtype=np.float64)
                self.get_logger().info(f"forward pass: {time.time() - t0:.3f}s")

                # safety check on actual joint positions from /joint_states
                # if not self.safety_check():
                #     self.get_logger().error("Joint limits exceeded — truncating episode.")
                #     truncated = True
                #     break

                targets = self.chunk_to_targets(action_chunk)

                new_stamp = primary.header.stamp if isinstance(primary, Image) else None
                frame_is_new = (new_stamp is not None) and (new_stamp != prev_stamp)
                self._last_img_stamp = new_stamp

                for i in range(targets.shape[0]):
                    # pace to step_duration (10 Hz)
                    elapsed = time.time() - last_tstep
                    if elapsed < step_duration:
                        time.sleep(step_duration - elapsed)
                    last_tstep = time.time()

                    if not self.joints_in_envelope():
                        self.get_logger().error(
                            f"Joints left demo envelope (±{ood_margin} rad): {self._current_joints} — ending episode, holding position."
                        )
                        self.set_desired_joints(self._current_joints)
                        truncated = True
                        break

                    joints_desired = targets[i]
                    self.get_logger().info(f"Desired Joints: {joints_desired}\nCurrent Joints: {self._current_joints}\nNew Frame:{frame_is_new}")

                    # with open("src/Franka_Bridge/joint_states_log_PepperRight.txt", "a") as file:
                    #     file.write(f"Desired Joints: {joints_desired}, Current Joints: {self._current_joints}, New Frame:{frame_is_new}\n")

                    self.set_ramp_target(joints_desired, duration=step_duration)

                    self.send_gripper(float(action_chunk[i, 7]))

                    step_count += 1
                    if step_count >= max_timesteps:
                        break

                if truncated:
                    break

            # Episode ended: hold position
            self.toggle_servo(start=False)

            status = (
                "truncated (out of bounds)"
                if truncated
                else f"completed ({max_timesteps} steps)"
            )
            self.get_logger().info(f"Episode {status}.")


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = OctoFrankaBridge()

    # continously running the executor in background to ensure it's firing @ 20 Hz for gello
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        node.load_model()
        node.start_controller()
        node.go_home()
        node.open_gripper()
        node.get_logger().info("home!!")

        node.main()
    finally:
        node.stop_controller()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    app.run(main)
