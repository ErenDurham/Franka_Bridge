"""
Moves robot with joint states 
Moves robot wrist in a sinusoidal motion
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time


TOPIC = "/gello/joint_states"
MSG = "sensor_msgs/msg/JointState"


class JointStatePublisher(Node):
    def __init__(self):
        super().__init__("gripper_joint_publisher")

        self.publisher = self.create_publisher(JointState, "/gello/joint_states", 10)

        timer_period = 1.0 / 20.0  # 20Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]

        self.base_positions = [
            -0.06367574853429891,
            0.22938613026211163,
            0.011507665397764821,
            -1.2008130054749508,
            0.1680173063454642,
            1.515181335577131,
            -0.9511231196761174,
        ]

        self.amplitude_joint6 = 0.2
        self.amplitude_joint7 = 0.3
        self.frequency = 0.5

        self.start_time = time.time()

    def timer_callback(self):
        msg = JointState()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        t = time.time() - self.start_time

        current_positions = list(self.base_positions)

        current_positions[5] += self.amplitude_joint6 * math.sin(
            2 * math.pi * self.frequency * t
        )
        current_positions[6] += self.amplitude_joint7 * math.cos(
            2 * math.pi * self.frequency * t
        )

        msg.position = current_positions
        msg.velocity = []
        msg.effort = []

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
