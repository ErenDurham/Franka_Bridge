"""
Previous code that I wrote to control gripper and connecting to an arduino
Using this file as reference for how to use gripper
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
import serial
import time


# Topic published by the Gello controller: a Float32 in [0.0, 1.0] where
# 0.0 = fully closed and 1.0 = fully open.
DEFAULT_GRIPPER_COMMAND_TOPIC = "gripper/gripper_client/target_gripper_width_percent"


# Topic this node publishes: a Bool that is True when the gripper is open
# and False when it is closed.
DEFAULT_GRIPPER_STATE_TOPIC = "gripper/gripper_client/gripper_is_open"


# Gripper width percent at or above which the gripper is considered open.
DEFAULT_GRIPPER_OPEN_THRESHOLD = 0.5


DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_SERIAL_BAUDRATE = 9600


class GripperStatePublisher(Node):
    """ROS2 node that reads Gello gripper commands and publishes an open/closed boolean."""

    def __init__(self):
        super().__init__("gripper_state_publisher")
        self.get_logger().info("Starting Gripper State Publisher")

        # Arduino Set up
        self.arduino = None

        try:
            self.arduino = serial.Serial(
                port=DEFAULT_SERIAL_PORT,
                baudrate=DEFAULT_SERIAL_BAUDRATE,
                timeout=1,
            )

            time.sleep(2)  # wait for the Arduino to respond
            self.get_logger().info(f"Opened serial port {DEFAULT_SERIAL_PORT}")

        except serial.SerialException as e:
            self.get_logger().error(
                f"Could not open serial port {DEFAULT_SERIAL_PORT}: {e}. "
                "Continuing without Arduino output."
            )

        # Subscribe to the Gello gripper width topic.
        self.create_subscription(
            Float32, DEFAULT_GRIPPER_COMMAND_TOPIC, self.gripper_state_callback, 10
        )

        # Publisher that broadcasts whether the gripper is open (True) or closed (False).
        self.gripper_is_open_publisher = self.create_publisher(
            Bool, DEFAULT_GRIPPER_STATE_TOPIC, 10
        )

        # Tracks the last published state to avoid re-publishing when unchanged.
        self._last_gripper_is_open = None

        self.get_logger().info("Gripper state publisher ready")

    def gripper_state_callback(self, msg):
        """Receive a Gello gripper width percent and publish open/closed state on transition."""

        is_open = msg.data >= DEFAULT_GRIPPER_OPEN_THRESHOLD  # bool to check if open

        if is_open != self._last_gripper_is_open:
            self._last_gripper_is_open = is_open
            self.gripper_is_open_publisher.publish(Bool(data=is_open))
            self._write_to_arduino(b"1" if is_open else b"0")  # 1 = open, 0 = closed
            print("GRIPPER command", is_open)

    def _write_to_arduino(self, payload):
        """Write a byte to the Arduino"""

        if self.arduino is None:
            return
        try:
            self.arduino.write(payload)
        except serial.SerialException as e:
            self.get_logger().warn(f"Failed to write to Arduino: {e}")

    def destroy_node(self):
        if self.arduino is not None and self.arduino.is_open:
            self.arduino.close()
            self.get_logger().info("Closed serial port")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GripperStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
