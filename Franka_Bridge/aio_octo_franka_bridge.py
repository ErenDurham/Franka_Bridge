"""
Written by Eren D.
Code to bridge the octo model and the Franka robot

Utilizing aiofranka FrankaRemoteController with OSC to control the robot through EE pose targets
"""

import math
import time
import numpy as np
from functools import partial
from octo.utils.train_callbacks import supply_rng
import jax
import jax.numpy as jnp
import cv2
import rclpy
from absl import flags
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from octo.model.octo_model import OctoModel
import click
from aiofranka import FrankaRemoteController

FLAGS = flags.FLAGS

TOPIC_IMAGE_PRIMARY = "/camera/primary/image_raw"
TOPIC_IMAGE_WRIST = "/camera/wrist/image_raw"
TOPIC_CURRENT_POSE = "/franka_robot_state_broadcaster/current_pose"

STEP_DURATION = 0.1  # (10 Hz) step size
MAX_TIMESTEPS = 200  # limits thinking time

# Gripper controls
GRIPPER_OPEN_WIDTH = 0.08  # meters
GRIPPER_CLOSE_WIDTH = 0.0
GRIPPER_SPEED = 0.1  # m/s
GRIPPER_FORCE = 10.0  # N
GRIPPER_THRESHOLD = 0.5  # action[-1] > threshold → open, else close

# Safety bounds
# [lower] [upper] limits. [x, y, z, yaw, grip]
# TODO: obtain yaw and grip values
WORKSPACE_BOUNDS = [
    [0.1516, -0.4215, 0.0097, -1.57, 0],
    [0.6844, 0.4193, 0.7058, 1.57, 0],
]

# Where the trained model weights live, and which training step to load
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
        self._current_pose = None
        self.model = None

        # Subscribers

        # subscribe to primary and wrist camera images
        self.create_subscription(Image, TOPIC_IMAGE_PRIMARY, self._cb_image_primary, 10)
        self.create_subscription(Image, TOPIC_IMAGE_WRIST, self._cb_image_wrist, 10)

        # get EE pose for state readback and proprio
        self.create_subscription(PoseStamped, TOPIC_CURRENT_POSE, self._cb_pose, 10)

        # aiofranka controller
        self._controller = FrankaRemoteController()
        self._ee_desired = None  # 4x4 desired EE transform, initialized after controller starts

    def _cb_image_primary(self, msg: Image) -> None:
        # store latest primary image
        self._image_primary = msg

    def _cb_image_wrist(self, msg: Image) -> None:
        # store latest wrist image
        self._image_wrist = msg

    def _cb_pose(self, msg: PoseStamped) -> None:
        # store latest EE pose
        self._current_pose = msg

    def load_model(self) -> None:
        self.model = OctoModel.load_pretrained(
            FLAGS.checkpoint_weights_path,
            FLAGS.checkpoint_step,
        )

    def start_controller(self) -> None:
        self._controller.start()

    def stop_controller(self) -> None:
        self._controller.stop()

    def toggle_servo(self, start=True):
        """
        Configures the controller, gains, and pose
        """
        if start:
            self._controller.switch("osc")

            # Task-space gains [x, y, z, roll, pitch, yaw]
            self._controller.ee_kp = np.array([300, 300, 300, 1000, 1000, 1000])
            self._controller.ee_kd = np.ones(6) * 10.0
            
            # Null-space gains (keeps robot away from joint limits)
            self._controller.null_kp = np.ones(7) * 10.0
            self._controller.null_kd = np.ones(7) * 1.0

            # Seed desired pose from current state so OSC engages without a jump
            self._ee_desired = self._controller.state['ee'].copy()
            self.get_logger().info("aiofranka OSC controller started")
        else:
            self.get_logger().info("aiofranka episode stopped, holding position")

    def go_home(self) -> None:
        """Move to home joint configuration via aiofranka joint-space move."""
        self._controller.move([0, 0, 0.0, -1.57079, 0, 1.57079, -0.7853])
        self.get_logger().info("go_home complete")

    def _ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        """Convert a ROS Image message to a (H, W, 3) uint8 numpy array."""
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        return arr[:, :, :3]

    # TODO: Finish gripper stuff
    def open_gripper(self) -> None:
        self.get_logger().warn("open_gripper: not set up yet :(")

    def publish_twist(self, ee_desired: np.ndarray) -> None:
        self._controller.set("ee_desired", ee_desired)

    # TODO: Finish gripper stuff
    def send_gripper(self, _gripper_value: float) -> None:
        self.get_logger().warn("send_gripper: not set up yet :(")

    def read_state(self) -> dict:
        ee = self._controller.state["ee"]
        p = ee[:3, 3]
        return {
            "ee": ee,
            "proprio": np.array(
                [p[0], p[1], p[2], *ee[:3, :3].flatten()[:4]], dtype=np.float32
            ),
        }

    def safety_check(self) -> bool:
        ee = self._controller.state["ee"]
        x, y, z = ee[0, 3], ee[1, 3], ee[2, 3]
        lo, hi = WORKSPACE_BOUNDS
        return lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and lo[2] <= z <= hi[2]

    def grab_image(self, img_size):
        # return (primary, wrist) or null obs if images unavailable
        # images missing -> print "Image not found", return null
        if self._image_primary is None or self._image_wrist is None:
            self.get_logger().info("Images not found. Using null images....")
            self._image_primary = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            self._image_wrist = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        return self._image_primary, self._image_wrist

    def convert_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert a 7-vector action [dx,dy,dz,droll,dpitch,dyaw,grip] to ee_desired update + gripper scalar."""
        # Apply position delta to ee_desired
        self._ee_desired[0, 3] += float(action[0])
        self._ee_desired[1, 3] += float(action[1])
        self._ee_desired[2, 3] += float(action[2])

        # Apply orientation delta (ZYX Euler) to the rotation part of ee_desired
        droll, dpitch, dyaw = float(action[3]), float(action[4]), float(action[5])
        cr, sr = math.cos(droll), math.sin(droll)
        cp, sp = math.cos(dpitch), math.sin(dpitch)
        cy, sy = math.cos(dyaw), math.sin(dyaw)
        R_delta = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr               ],
        ])
        self._ee_desired[:3, :3] = R_delta @ self._ee_desired[:3, :3]

        return self._ee_desired.copy(), float(action[6])


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
            return actions[0]

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

                # flush ROS callbacks to update image + pose buffers
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
                obs_history.append(
                    {"image_primary": img_primary, "image_wrist": img_wrist}
                )
                if len(obs_history) > window_size:
                    obs_history.pop(0)

                # pad to window_size by repeating the earliest frame
                pad = obs_history[0:1] * (window_size - len(obs_history))
                windowed = pad + obs_history
                obs = {
                    "image_primary": np.stack([o["image_primary"] for o in windowed]),
                    "image_wrist": np.stack([o["image_wrist"] for o in windowed]),
                }

                # model inference
                t0 = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                self.get_logger().info(f"forward pass: {time.time() - t0:.3f}s")

                # ee_desired update -> controller, gripper -> aiofranka gripper
                ee_desired, gripper_val = self.convert_action(action)
                self.publish_twist(ee_desired)
                self.send_gripper(gripper_val)

                # safety check on latest EE pose from aiofranka state
                if not self.safety_check():
                    self.get_logger().warn(
                        "Workspace bounds exceeded — truncating episode."
                    )
                    truncated = True
                    break

            # Episode ended: hold position and stop controller
            self.toggle_servo(start=False)

            status = (
                "truncated (out of bounds)"
                if truncated
                else f"completed ({MAX_TIMESTEPS} steps)"
            )
            self.get_logger().info(f"Episode {status}.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OctoFrankaBridge()
    try:
        # load model weights
        node.load_model()

        # start the aiofranka server, then home in joint-space before switching to OSC
        node.start_controller()
        node.go_home()
        node.open_gripper()

        # run inference episodes
        node.main()
    finally:
        node.stop_controller()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
