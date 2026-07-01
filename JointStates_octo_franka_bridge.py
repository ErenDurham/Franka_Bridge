"""
Code to bridge the octo model and the Franka robot
Utilizing sensor_msgs/JointState published to /gello/joint_states to command joint positions
"""

import time
import numpy as np
from functools import partial
from octo.utils.train_callbacks import supply_rng
import jax
import jax.numpy as jnp
import cv2
import rclpy
from absl import app, flags
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from octo.model.octo_model import OctoModel
import click

FLAGS = flags.FLAGS

TOPIC_IMAGE_PRIMARY = "/camera/primary/image_raw"
TOPIC_IMAGE_WRIST = "/camera/wrist/image_raw"
TOPIC_JOINT_STATES = "/joint_states"      # robot's actual state (franka joint_state_broadcaster)
TOPIC_JOINT_CMD = "/gello/joint_states"   # desired joint positions (GELLO middleware input)

STEP_DURATION = 0.1  # (10 Hz) step size
MAX_TIMESTEPS = 200  

# Gripper controls
GRIPPER_OPEN_WIDTH = 0.08  # meters
GRIPPER_CLOSE_WIDTH = 0.0
GRIPPER_SPEED = 0.1  # m/s
GRIPPER_FORCE = 10.0  # N
GRIPPER_THRESHOLD = 0.5  # action[-1] > threshold -> open, else close

# FR3 software joint position limits from franka_description/robots/fr3/joint_limits.yaml
JOINT_LIMITS = [
    [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800,  0.8521, -2.6895],
    [ 2.3093,  1.5133,  2.4937, -0.4461,  2.4800,  4.2094,  2.6895],
]

FR3_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

HOME_JOINTS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

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
        self._current_joints = None  # 7-element array from /joint_states
        self._joints_desired = None  # desired joint positions, seeded after go_home
        self.model = None

        # Subscribers

        # subscribe to primary and wrist camera images
        self.create_subscription(Image, TOPIC_IMAGE_PRIMARY, self._cb_image_primary, 10)
        self.create_subscription(Image, TOPIC_IMAGE_WRIST, self._cb_image_wrist, 10)

        # get joint positions for state readback and safety checks
        self.create_subscription(JointState, TOPIC_JOINT_STATES, self._cb_joint_states, 10)

        # Publisher: desired joint states -> GELLO middleware -> robot
        self._joint_pub = self.create_publisher(JointState, TOPIC_JOINT_CMD, 10)

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

    def load_model(self) -> None:
        self.model = OctoModel.load_pretrained(
            FLAGS.checkpoint_weights_path,
            FLAGS.checkpoint_step,
        )
        keys = list(self.model.dataset_statistics.keys())
        self.get_logger().info(f"Dataset statistics keys: {keys}")
        if "2026_Data" not in self.model.dataset_statistics:
            raise KeyError(f"'2026_Data' not in dataset_statistics. Available: {keys}")

    def start_controller(self) -> None:
        # spin until first joint state message arrives
        while self._current_joints is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info("Waiting for joint states...")
        self.get_logger().info("Joint states received, controller ready.")

    def stop_controller(self) -> None:
        self.get_logger().info("Episode stopped, holding last joint position.")

    def toggle_servo(self, start=True):
        if start:
            # seed desired joints from current state so controller engages without a jump
            rclpy.spin_once(self, timeout_sec=0.1)
            self._joints_desired = self._current_joints.copy()
            self.get_logger().info("Joint controller started")
        else:
            self.get_logger().info("Joint controller stopped, holding position")

    def go_home(self) -> None:
        """Move to home joint configuration."""
        end_time = time.time() + 5.0
        while time.time() < end_time:
            self.publish_joint_state(HOME_JOINTS)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)
        self.get_logger().info("go_home complete")

    def _ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        """Convert a ROS Image message to a (H, W, 3) uint8 numpy array."""
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        return arr[:, :, :3]

    # TODO: Finish gripper stuff
    def open_gripper(self) -> None:
        self.get_logger().warn("open_gripper: not set up yet :(")

    def publish_joint_state(self, joints_desired: np.ndarray) -> None:
        """Publish desired joint positions as JointState to /gello/joint_states."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = FR3_JOINTS
        msg.position = joints_desired.tolist()
        msg.velocity = []
        msg.effort = []
        self._joint_pub.publish(msg)

    # TODO: Finish gripper stuff
    def send_gripper(self, _gripper_value: float) -> None:
        self.get_logger().warn("send_gripper: not set up yet :(")

    def read_state(self) -> dict:
        q = self._current_joints if self._current_joints is not None else np.zeros(7)
        return {
            "joints": q,
            "proprio": q.astype(np.float32),
        }

    def safety_check(self) -> bool:
        if self._current_joints is None:
            return True
        lo, hi = JOINT_LIMITS
        return all(lo[i] <= self._current_joints[i] <= hi[i] for i in range(7))

    def grab_image(self, img_size):
        # return (primary, wrist) or null obs if images unavailable
        if self._image_primary is None or self._image_wrist is None:
            self.get_logger().info("Images not found. Using null images....")
            self._image_primary = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            self._image_wrist = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            self.get_logger().info("Images FOUND!!! :D")
        return self._image_primary, self._image_wrist

    def convert_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert model action [dq1..dq7, grip] to joints_desired update + gripper scalar."""
        # NOTE: change = to += if dataset outputs deltas.
        lo = np.array(JOINT_LIMITS[0])
        hi = np.array(JOINT_LIMITS[1])
        self._joints_desired = action[:7]
        self._joints_desired = np.clip(self._joints_desired, lo, hi)
        return self._joints_desired.copy(), float(action[7])

    # Main loop
    def main(self, im_size: int = 256, window_size: int = 2) -> None:
        """
        Runs inference episode
        Builds function, obtains task/goal image, and runs the inference
        """

        # Build the inference function
        # TODO: Change the path to the dataset
        def _sample_actions(pretrained_model, observations, tasks, rng):
            observations = jax.tree_map(lambda x: x[None], observations)
            actions = pretrained_model.sample_actions(
                observations,
                tasks,
                rng=rng,
                unnormalization_statistics=pretrained_model.dataset_statistics[
                    "2026_Data"
                ]["action"],
            )
            # sample_actions returns (batch, action_horizon, action_dim);
            # [0, 0] selects first batch item, first predicted timestep -> (action_dim,)
            return actions[0, 0]

        policy_fn = supply_rng(partial(_sample_actions, self.model))

        goal_image = jnp.zeros((im_size, im_size, 3), dtype=np.uint8)
        goal_instruction = ""

        while True:
            # Goal selection
            modality = click.prompt(
                "Language or goal image?", type=click.Choice(["l", "g"])
            )

            # Goal image chosen
            if modality == "g":
                # Check if new goal image is needed
                if click.confirm("Take a new goal?", default=True):
                    input("Move arm to goal pose, then press [Enter] to capture.")
                    rclpy.spin_once(self, timeout_sec=0.5)
                    primary, _ = self.grab_image(im_size)
                    if isinstance(primary, Image):
                        raw = self._ros_image_to_numpy(primary)  # format image
                        goal_image = cv2.resize(raw, (im_size, im_size))
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
                    (im_size, im_size, 3), dtype=np.uint8
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

                # flush ROS callbacks to update image + joint state buffers
                rclpy.spin_once(self, timeout_sec=0.02)

                # build observations from latest camera frames
                primary, wrist = self.grab_image(im_size)
                img_primary = (
                    cv2.resize(self._ros_image_to_numpy(primary), (im_size, im_size))
                    if isinstance(primary, Image)
                    else primary
                )
                img_wrist = (
                    cv2.resize(self._ros_image_to_numpy(wrist), (im_size, im_size))
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
                    # False = padded frame, True = real frame; shape (window_size,)
                    # _sample_actions adds batch dim via jax.tree_map -> (1, window_size)
                    "timestep_pad_mask": np.array(
                        [False] * pad_count + [True] * len(obs_history), dtype=bool
                    ),
                }

                # model inference
                t0 = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                self.get_logger().info(f"forward pass: {time.time() - t0:.3f}s")

                # safety check on actual joint positions from /joint_states
                if not self.safety_check():
                    self.get_logger().warn(
                        "Joint limits exceeded. truncating episode."
                    )
                    truncated = True
                    break

                # joints_desired update -> /gello/joint_states, gripper -> placeholder
                joints_desired, gripper_val = self.convert_action(action)
                self.publish_joint_state(joints_desired)
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
    try:
        node.load_model()
        node.start_controller()
        node.go_home()
        node.open_gripper()
        node.main()
    finally:
        node.stop_controller()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    app.run(main)
