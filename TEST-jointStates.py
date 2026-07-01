"""
Code to test the bridge the octo model and the Franka robot WITHOUT Octo to ensure logic is right

Utilizing sensor_msgs/JointState published to /gello/joint_states to command joint positions
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32

TOPIC_IMAGE_PRIMARY = "/camera/primary/image_raw"
TOPIC_IMAGE_WRIST = "/camera/wrist/image_raw"
TOPIC_JOINT_STATES = (
    "/joint_states"  # robot's actual state (franka joint_state_broadcaster)
)
TOPIC_JOINT_CMD = (
    "/gello/joint_states"  # desired joint positions (GELLO middleware input)
)

# Topic published by the Gello controller: a Float32 in [0.0, 1.0] where
# 0.0 = fully closed and 1.0 = fully open.
DEFAULT_GRIPPER_COMMAND_TOPIC = "gripper/gripper_client/target_gripper_width_percent"


# Topic this node publishes: a Bool that is True when the gripper is open
# and False when it is closed.
DEFAULT_GRIPPER_STATE_TOPIC = "gripper/gripper_client/gripper_is_open"

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

HOME_JOINTS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Test target: joint 4 lowered 0.3 rad from home (arm drops slightly)
TEST_JOINTS = np.array([0.0, 0.0, 0.0, -1.87079, 0.0, 1.57079, -0.7853])


class OctoFrankaBridge(Node):
    def __init__(self):
        super().__init__("octo_franka_bridge")

        # image and state buffers
        self._image_primary = None
        self._image_wrist = None
        self._current_joints = None  # 7-element array from /joint_states
        self._joints_desired = None  # desired joint positions, seeded after go_home

        # Subscribers

        # subscribe to primary and wrist camera images
        self.create_subscription(Image, TOPIC_IMAGE_PRIMARY, self._cb_image_primary, 10)
        self.create_subscription(Image, TOPIC_IMAGE_WRIST, self._cb_image_wrist, 10)

        # get joint positions for state readback and safety checks
        self.create_subscription(
            JointState, TOPIC_JOINT_STATES, self._cb_joint_states, 10
        )

        # Publishers
        self._joint_pub = self.create_publisher(JointState, TOPIC_JOINT_CMD, 10)
        self._gripper_pub = self.create_publisher(
            Float32, DEFAULT_GRIPPER_COMMAND_TOPIC, 10
        )

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
        pct = float(np.clip(gripper_value, 0.0, 1.0))
        self._gripper_pub.publish(Float32(data=pct))

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
        return self._image_primary, self._image_wrist

    def convert_action(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert model action [dq1..dq7, grip] to joints_desired update + gripper scalar."""
        # NOTE: change += to = if dataset outputs absolute positions instead of deltas.
        lo = np.array(JOINT_LIMITS[0])
        hi = np.array(JOINT_LIMITS[1])
        self._joints_desired += action[:7]
        self._joints_desired = np.clip(self._joints_desired, lo, hi)
        return self._joints_desired.copy(), float(action[7])

    # Main loop
    def main(self) -> None:
        """
        Runs test episode
        Commands TEST_JOINTS for MAX_TIMESTEPS steps, then loops.
        """

        while True:
            input("Press [Enter] to start.")
            self.toggle_servo(start=True)

            ########## Control loop  ###############
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

                # safety check on actual joint positions from /joint_states
                if not self.safety_check():
                    self.get_logger().warn("Joint limits exceeded. truncating episode.")
                    truncated = True
                    break

                # command the test position directly
                self.publish_joint_state(TEST_JOINTS)
                self.send_gripper(0.0)

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
        node.start_controller()
        node.go_home()
        node.close_gripper()
        node.main()
    finally:
        node.stop_controller()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
