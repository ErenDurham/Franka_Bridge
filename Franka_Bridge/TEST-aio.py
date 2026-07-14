"""
Written by Eren D.
Code to control the Franka robot via aiofranka FrankaRemoteController with OSC.
Moves the end-effector to a target pose within the safe workspace bounds.
"""

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from aiofranka import FrankaRemoteController

TOPIC_IMAGE_PRIMARY = "/franka/image_primary"
TOPIC_IMAGE_WRIST = "/franka/image_wrist"
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

# Midpoint of workspace used as the default move target
TARGET_POSITION = np.array([
    (WORKSPACE_BOUNDS[0][0] + WORKSPACE_BOUNDS[1][0]) / 2.0,
    (WORKSPACE_BOUNDS[0][1] + WORKSPACE_BOUNDS[1][1]) / 2.0,
    (WORKSPACE_BOUNDS[0][2] + WORKSPACE_BOUNDS[1][2]) / 2.0,
])

POSITION_TOLERANCE = 0.005  # 5 mm


class FrankaBridge(Node):
    def __init__(self):
        super().__init__("franka_bridge")

        # image and state buffers
        self._image_primary = None
        self._image_wrist = None
        self._current_pose = None

        # Subscribers
        self.create_subscription(Image, TOPIC_IMAGE_PRIMARY, self._cb_image_primary, 10)
        self.create_subscription(Image, TOPIC_IMAGE_WRIST, self._cb_image_wrist, 10)
        self.create_subscription(PoseStamped, TOPIC_CURRENT_POSE, self._cb_pose, 10)

        # aiofranka controller
        self._controller = FrankaRemoteController()
        self._ee_desired = (
            None  # 4x4 desired EE transform, initialized after controller starts
        )

    def _cb_image_primary(self, msg: Image) -> None:
        # store latest primary image
        self._image_primary = msg

    def _cb_image_wrist(self, msg: Image) -> None:
        # store latest wrist image
        self._image_wrist = msg

    def _cb_pose(self, msg: PoseStamped) -> None:
        # store latest EE pose
        self._current_pose = msg

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
            self._ee_desired = self._controller.state["ee"].copy()
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

    def open_gripper(self) -> None:
        self.get_logger().warn("open_gripper: aiofranka has no gripper API — skipping")

    def publish_twist(self, ee_desired: np.ndarray) -> None:
        self._controller.set("ee_desired", ee_desired)

    def send_gripper(self, _gripper_value: float) -> None:
        self.get_logger().warn("send_gripper: aiofranka has no gripper API — skipping")

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
        R_delta = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
        self._ee_desired[:3, :3] = R_delta @ self._ee_desired[:3, :3]

        return self._ee_desired.copy(), float(action[6])

    # Main loop
    def main(self) -> None:
        """Move the end-effector to TARGET_POSITION within the safe workspace bounds."""
        self.toggle_servo(start=True)

        if self._ee_desired is None:
            self.get_logger().error("ee_desired not initialized. aborting.")
            return

        # Build target ee transform: keep current orientation, update position
        target_ee = self._ee_desired.copy()
        target_ee[0, 3] = TARGET_POSITION[0]
        target_ee[1, 3] = TARGET_POSITION[1]
        target_ee[2, 3] = TARGET_POSITION[2]

        self.get_logger().info(
            f"Moving to target: x={TARGET_POSITION[0]:.3f}, y={TARGET_POSITION[1]:.3f}, z={TARGET_POSITION[2]:.3f}"
        )

        last_tstep = time.time()
        reached = False

        for _timestep in range(MAX_TIMESTEPS):
            # pace to STEP_DURATION (10 Hz)
            elapsed = time.time() - last_tstep
            if elapsed < STEP_DURATION:
                time.sleep(STEP_DURATION - elapsed)
            last_tstep = time.time()

            # flush ROS callbacks to update image + pose buffers
            rclpy.spin_once(self, timeout_sec=0.02)

            # safety check on latest EE pose from aiofranka state
            if not self.safety_check():
                self.get_logger().warn(
                    "Workspace bounds exceeded. stopping episode."
                )
                break

            self.publish_twist(target_ee)

            # check distance to target
            ee = self._controller.state["ee"]
            pos = ee[:3, 3]
            dist = np.linalg.norm(pos - TARGET_POSITION)
            if dist < POSITION_TOLERANCE:
                self.get_logger().info(
                    f"Target reached (dist={dist*1000:.1f} mm)."
                )
                reached = True
                break

        self.toggle_servo(start=False)
        status = "reached" if reached else "not reached within step limit"
        self.get_logger().info(f"Episode complete — target {status}.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrankaBridge()
    try:
        # start the aiofranka server, then home in joint-space before switching to OSC
        node.start_controller()
        node.go_home()
        node.open_gripper()

        # move to target position
        node.main()
    finally:
        node.stop_controller()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
