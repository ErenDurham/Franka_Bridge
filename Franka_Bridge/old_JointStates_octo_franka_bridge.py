"""
Written by Eren D.
Code to bridge the octo model and the Franka robot
Utilizing sensor_msgs/JointState published to /gello/joint_states to command joint positions

To run:
conda activate octo
python3 src/Franka_Bridge/old_JointStates_octo_franka_bridge.py --checkpoint_weights_path="/home/faro/octo_fr3/checkpoints/Fixed_Data_Checkpoint/octo_fr3_finetune/experiment_20260715_191603" --checkpoint_step="50000"
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
from absl import app, flags
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from rclpy.action import ActionClient
from franka_msgs.action import Move as GripperMove
from octo.model.octo_model import OctoModel
import click

FLAGS = flags.FLAGS

TOPIC_IMAGE_PRIMARY = "/camera/primary/image_raw"
TOPIC_IMAGE_WRIST = "/camera/wrist/image_raw"
TOPIC_JOINT_STATES = (
    "/joint_states"  # robot's actual state
)
TOPIC_JOINT_CMD = (
    "/gello/joint_states"  # desired joint positions
)
TOPIC_GRIPPER_STATE = "/franka_gripper/joint_states"  # actual finger positions

IMG_SIZE_PRIMARY = 256
IMG_SIZE_WRIST = 128
WINDOW_SIZE = 1

# 0.0 = fully closed and 1.0 = fully open.
DEFAULT_GRIPPER_COMMAND_TOPIC = "/gripper/gripper_client/target_gripper_width_percent"

STEP_DURATION = 0.1  # (10 Hz) step size
MAX_TIMESTEPS = 1000
JOINT_SMOOTH_ALPHA = 0.5

# Gripper controls
GRIPPER_OPEN_WIDTH = 0.08  # meters
GRIPPER_CLOSE_WIDTH = 0.0
GRIPPER_SPEED = 0.1  # m/s
GRIPPER_FORCE = 10.0  # N
GRIPPER_THRESHOLD = 0.5  

# FR3 software joint position limits from franka_description/robots/fr3/joint_limits.yaml
JOINT_LIMITS = [
    [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895],
    [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895],
]

FR3_JOINTS = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

HOME_JOINTS = np.array([-0.046, 0.242, -0.123, -1.593, 0.100, 1.967, -0.903])

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)


class OctoFrankaBridge(Node):
    def __init__(self):
        super().__init__("octo_franka_bridge")

        # image and state buffers
        self._image_primary = None
        self._image_wrist = None
        self._current_joints = None
        self._gripper_width = 0.0
        self._gripper_is_open = None
        self.model = None

        # Subscribers

        # subscribe to primary and wrist camera images
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, TOPIC_IMAGE_PRIMARY, self._cb_image_primary, image_qos)
        self.create_subscription(Image, TOPIC_IMAGE_WRIST, self._cb_image_wrist, image_qos)

        # get joint positions
        self.create_subscription(
            JointState, TOPIC_JOINT_STATES, self._cb_joint_states, 10
        )

        # get gripper finger positions (width = sum of both fingers)
        self.create_subscription(
            JointState, TOPIC_GRIPPER_STATE, self._cb_gripper_state, 10
        )

        # Publishers
        self._joint_pub = self.create_publisher(JointState, TOPIC_JOINT_CMD, 10)
        self._grip_move_cli = ActionClient(self, GripperMove, "/franka_gripper/move")

        self._joints_desired = HOME_JOINTS.copy()
        self._joints_lock = threading.Lock()
        self._ramp_start = None
        self._ramp_target = None
        self._ramp_t0 = 0.0
        self._ramp_duration = 3.0
        self.create_timer(0.1, self._timer_publish_joints)

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
            idx = [msg.name.index(j) for j in FR3_JOINTS]
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
            FLAGS.checkpoint_weights_path,
            FLAGS.checkpoint_step,
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

    def start_controller(self) -> None:
        # spin until first joint state message arrives
        while self._current_joints is None:
            time.sleep(0.1)
            self.get_logger().info("Waiting for joint states...")
        self.get_logger().info("Joint states received, controller ready")

    def stop_controller(self) -> None:
        self.get_logger().info("Episode stopped, holding last joint position.")

    def toggle_servo(self, start=True):
        if start:
            if self._current_joints is not None:
                self.ramp_to(self._current_joints, duration=STEP_DURATION)
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
        Ramp to HOME_JOINTS over 3s.
        """
        self.ramp_to(HOME_JOINTS, duration=3.0)
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
        msg.name = FR3_JOINTS
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
        target_open = gripper_value > GRIPPER_THRESHOLD
        if target_open == self._gripper_is_open:
            return

        # find if topic is available
        if not self._grip_move_cli.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Gripper not available!! :(")
            return

        goal = GripperMove.Goal()
        goal.width = GRIPPER_OPEN_WIDTH if target_open else GRIPPER_CLOSE_WIDTH
        goal.speed = GRIPPER_SPEED

        future = self._grip_move_cli.send_goal_async(goal)
        self._gripper_is_open = target_open

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
        lo, hi = JOINT_LIMITS
        return all(lo[i] <= self._current_joints[i] <= hi[i] for i in range(7))

    def grab_image(self):
        # return (primary, wrist), substituting null images per-camera if unavailable
        if self._image_primary is None:
            self.get_logger().info("Primary image not found. Using null image....")
            primary = np.zeros((IMG_SIZE_PRIMARY, IMG_SIZE_PRIMARY, 3), dtype=np.uint8)
        else:
            primary = self._image_primary
        if self._image_wrist is None:
            self.get_logger().info("Wrist image not found. Using null image....")
            wrist = np.zeros((IMG_SIZE_WRIST, IMG_SIZE_WRIST, 3), dtype=np.uint8)
        else:
            wrist = self._image_wrist
        return primary, wrist

    def convert_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Smooths the actions
        """

        lo = np.array(JOINT_LIMITS[0])
        hi = np.array(JOINT_LIMITS[1])
        target = np.clip(action[:7], lo, hi)

        with self._joints_lock:
            prev = self._joints_desired.copy()
        smoothed = JOINT_SMOOTH_ALPHA * target + (1.0 - JOINT_SMOOTH_ALPHA) * prev

        return smoothed, float(action[7])

    def convert_action_no_limits(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Smooths the actions without clipping to JOINT_LIMITS (Debug function)
        """

        target = action[:7]

        with self._joints_lock:
            prev = self._joints_desired.copy()
        smoothed = JOINT_SMOOTH_ALPHA * target + (1.0 - JOINT_SMOOTH_ALPHA) * prev

        return smoothed, float(action[7])

    # Main loop
    def main(self, window_size: int = WINDOW_SIZE) -> None:
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
            # [0, 0] selects first batch item, first predicted timestep -> (action_dim,)
            return actions[0, 0]

        policy_fn = supply_rng(partial(_sample_actions, self.model))

        goal_image = jnp.zeros((IMG_SIZE_PRIMARY, IMG_SIZE_PRIMARY, 3), dtype=np.uint8)
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
                        goal_image = cv2.resize(raw, (IMG_SIZE_PRIMARY, IMG_SIZE_PRIMARY))
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
                    (IMG_SIZE_PRIMARY, IMG_SIZE_PRIMARY, 3), dtype=np.uint8
                )  # empty goal image

            else:
                raise NotImplementedError()

            input("Press [Enter] to start.")
            self.toggle_servo(start=True)

            ########## Control loop  ###############
            
            obs_history: list[dict] = []
            last_tstep = time.time()
            truncated = False

            for _timestep in range(MAX_TIMESTEPS):
                # pace to STEP_DURATION (10 Hz)
                elapsed = time.time() - last_tstep
                if elapsed < STEP_DURATION:
                    time.sleep(STEP_DURATION - elapsed)
                last_tstep = time.time()

                # build observations from latest camera frames
                prev_stamp = getattr(self, "_last_img_stamp", None)

                primary, wrist = self.grab_image()
                img_primary = (
                    cv2.resize(self._ros_image_to_numpy(primary), (IMG_SIZE_PRIMARY, IMG_SIZE_PRIMARY))
                    if isinstance(primary, Image)
                    else primary
                )
                img_wrist = (
                    cv2.resize(self._ros_image_to_numpy(wrist), (IMG_SIZE_WRIST, IMG_SIZE_WRIST))
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

                # model inference
                t0 = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                self.get_logger().info(f"forward pass: {time.time() - t0:.3f}s")

                # safety check on actual joint positions from /joint_states
                # if not self.safety_check():
                #     self.get_logger().error("Joint limits exceeded — truncating episode.")
                #     truncated = True
                #     break

                # joints_desired update -> /gello/joint_states, gripper -> placeholder
                joints_desired, gripper_val = self.convert_action(action)

                new_stamp = primary.header.stamp if isinstance(primary, Image) else None
                frame_is_new = (new_stamp is not None) and (new_stamp != prev_stamp)
                self._last_img_stamp = new_stamp

                self.get_logger().info(f"Desired Joints: {joints_desired}\nCurrent Joints: {self._current_joints}\nNew Frame:{frame_is_new}")

                self.set_ramp_target(joints_desired, duration=1.0)

                self.send_gripper(gripper_val)

            
            # Episode ended: hold position
            self.toggle_servo(start=False)

            status = (
                "truncated (out of bounds)"
                if truncated
                else f"completed ({MAX_TIMESTEPS} steps)"
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


