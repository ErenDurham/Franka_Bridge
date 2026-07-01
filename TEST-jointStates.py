"""
Code to test the bridge the octo model and the Franka robot WITHOUT Octo to ensure logic is right

Utilizing sensor_msgs/JointState published to /gello/joint_states to command joint positions
"""

import threading
import time
import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from rclpy.action import ActionClient
from franka_msgs.action import Move as GripperMove

TOPIC_IMAGE_PRIMARY = "/camera/primary/image_raw"
TOPIC_IMAGE_WRIST = "/camera/wrist/image_raw"
TOPIC_JOINT_STATES = (
    "/joint_states"  # robot's actual state 
)
TOPIC_JOINT_CMD = (
    "/gello/joint_states"  # desired joint positions 
)

# Topic published by the Gello controller: a Float32 in [0.0, 1.0] where
# 0.0 = fully closed and 1.0 = fully open.
DEFAULT_GRIPPER_COMMAND_TOPIC = "/gripper/gripper_client/target_gripper_width_percent"

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

# drop a little below home
TEST_JOINTS = np.array([0.0, 0.0, 0.0, -1.27079, 0.0, 1.57079, -0.7853])


class OctoFrankaBridge(Node):
    def __init__(self):
        super().__init__("octo_franka_bridge")

        # image and state buffers
        self._image_primary = None
        self._image_wrist = None
        self._current_joints = None

        # Subscribers

        # subscribe to primary and wrist camera images
        self.create_subscription(Image, TOPIC_IMAGE_PRIMARY, self._cb_image_primary, 10)
        self.create_subscription(Image, TOPIC_IMAGE_WRIST, self._cb_image_wrist, 10)

        # get joint positions
        self.create_subscription(
            JointState, TOPIC_JOINT_STATES, self._cb_joint_states, 10
        )

        # Publishers
        self._joint_pub = self.create_publisher(JointState, TOPIC_JOINT_CMD, 10)
        self._grip_move_cli = ActionClient(self, GripperMove, "/franka_gripper/move")

        # Creating a timer to ensure there's continuous publishing of joint states
        self._joints_desired = HOME_JOINTS.copy()
        self._joints_lock = threading.Lock()
        self.create_timer(0.05, self._timer_publish_joints)

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

    def _timer_publish_joints(self) -> None:
        with self._joints_lock:
            joints = self._joints_desired.copy()
        self.publish_joint_state(joints)

    def set_desired_joints(self, joints: np.ndarray) -> None:
        with self._joints_lock:
            self._joints_desired = joints.copy()

    def start_controller(self) -> None:
        while self._current_joints is None:
            time.sleep(0.1)
            self.get_logger().info("Waiting for joint states...")
        self.get_logger().info("Joint states received, controller ready")

    def stop_controller(self) -> None:
        self.get_logger().info("Episode stopped, holding last joint position")

    def toggle_servo(self, start=True):
        if start:
            if self._current_joints is not None:
                self.set_desired_joints(self._current_joints)
            self.get_logger().info("Joint controller started")
        else:
            self.get_logger().info("Joint controller stopped, holding position")

    def go_home(self) -> None:
        """send to home and wait 5s"""
        self.set_desired_joints(HOME_JOINTS)
        time.sleep(5.0)
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
        """
        CLI cmd:
            ros2 action send_goal /franka_gripper/move franka_msgs/action/Move "{width: 0.08, speed: 0.1}"
            ros2 action send_goal /franka_gripper/move franka_msgs/action/Move "{width: 0.0, speed: 0.1}"
        """
        # find if topic is available
        if not self._grip_move_cli.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("franka_gripper/move action server not available!! :(")
            return

        goal = GripperMove.Goal()
        goal.width = float(np.clip(gripper_value, 0.0, 1.0)) * 0.08  
        goal.speed = 0.1

        future = self._grip_move_cli.send_goal_async(goal)

        self.get_logger().info("send_gripper complete")
        

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
        # NOTE: change = to += if dataset outputs deltas.
        lo = np.array(JOINT_LIMITS[0])
        hi = np.array(JOINT_LIMITS[1])
        self._joints_desired = action[:7]
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
                elapsed = time.time() - last_tstep
                if elapsed < STEP_DURATION:
                    time.sleep(STEP_DURATION - elapsed)
                last_tstep = time.time()

                if not self.safety_check():
                    self.get_logger().error("Joint limits exceeded — truncating episode.")
                    truncated = True
                    break

                self.set_desired_joints(TEST_JOINTS)
                self.get_logger().info(f"step {_timestep}: sent the test joints, actual={self._current_joints}")

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
        node.start_controller()
        node.go_home()
        node.get_logger().info("home!!")

        node.close_gripper()
        node.get_logger().info("Gripper closed!")
        node.main()
    finally:
        node.stop_controller()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()