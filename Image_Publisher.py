#!/usr/bin/env python3
"""
Written by: Makhtar N.

Image_Publisher.py — run on the camera machine, in a ROS2-sourced shell.

Streams full-resolution 1280x720 BGR frames from both RealSense cameras with
no resizing or colour conversion, publishes them as ROS2 Image messages, and
manages keypress-controlled rosbag recording and MP4 video saving per demo.

Each demo saves to its own folder:
  ~/pepper_robot_cam/captures/demo_001/
      demo_001_0.db3          ← rosbag data
      metadata.yaml           ← rosbag index
      primary.mp4             ← third-person camera video
      wrist.mp4               ← wrist camera video

Keypress controls:
  R — start recording a new demonstration
  S — stop recording and save
  Q — quit the node

Usage:
    source /opt/ros/<distro>/setup.bash
    export ROS_DOMAIN_ID=42
    python3 Image_Publisher.py
"""

import os
import sys
import tty
import termios
import threading
import subprocess
import signal
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

# ---------------- CONFIG ----------------
PRIMARY_SERIAL = "220422302020"   # D455 third-person
WRIST_SERIAL   = "919122070892"   # D435 wrist
STREAM_WIDTH   = 1280
STREAM_HEIGHT  = 720
STREAM_FPS     = 15
RATE_HZ        = 15


PUB_WIDTH      = 512
PUB_HEIGHT     = 288

CAPTURES_DIR   = Path.home() / "pepper_robot_cam" / "captures"
# ----------------------------------------

# MP4 writer settings
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


# ── helpers ──────────────────────────────────────────────────────────────────

def start_cam(serial: str) -> rs.pipeline:
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(
        rs.stream.color,
        STREAM_WIDTH, STREAM_HEIGHT,
        rs.format.bgr8, STREAM_FPS,
    )
    pipe.start(cfg)
    return pipe


def grab_raw_frame(pipe: rs.pipeline) -> np.ndarray:
    """Return the raw 1280x720 BGR frame as-is — no conversion, no resize."""
    frame = pipe.wait_for_frames().get_color_frame()
    return np.ascontiguousarray(np.asanyarray(frame.get_data()))


def numpy_to_image_msg(bgr: np.ndarray, stamp, frame_id: str) -> Image:
    msg                 = Image()
    msg.header.stamp    = stamp
    msg.header.frame_id = frame_id
    msg.height          = bgr.shape[0]
    msg.width           = bgr.shape[1]
    msg.encoding        = "bgr8"
    msg.is_bigendian    = 0
    msg.step            = msg.width * 3
    msg.data            = bgr.tobytes()
    return msg


def next_demo_path() -> Path:
    """Return the next available demo folder, e.g. ~/pepper_robot_cam/captures/demo_004."""
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CAPTURES_DIR.glob("demo_*"))
    next_idx = len(existing) + 1
    return CAPTURES_DIR / f"demo_{next_idx:03d}"


def get_keypress() -> str:
    """Read a single keypress from stdin without waiting for Enter.
    Always restores terminal state even if an error occurs."""
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch.lower()
    except Exception:
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, old)


# ── node ─────────────────────────────────────────────────────────────────────

class ImagePublisher(Node):
    """
    Publishes full-resolution BGR frames, records rosbags, and saves MP4
    videos per demonstration.

    Controls:
      R — start a new demo recording
      S — stop and save current recording
      Q — quit
    """

    def __init__(self):
        super().__init__("Image_Publisher")

        # RELIABLE so rosbag record captures every frame (a reliable publisher
        # is still compatible with best-effort subscribers like the bridge).
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.primary_pipe = start_cam(PRIMARY_SERIAL)
        self.wrist_pipe   = start_cam(WRIST_SERIAL)

        self.pub_primary = self.create_publisher(
            Image, "/camera/primary/image_raw", qos)
        self.pub_wrist   = self.create_publisher(
            Image, "/camera/wrist/image_raw",   qos)

        self.create_timer(1.0 / RATE_HZ, self.tick)

        # rosbag state
        # self._bag_proc:   subprocess.Popen | None = None
        # self._bag_log                             = None
        self._demo_path:  Path | None             = None
        self._demo_count: int                     = 0
        self._recording:  bool                    = False

        # video writer state — one writer per camera per demo
        self._writer_primary: cv2.VideoWriter | None = None
        self._writer_wrist:   cv2.VideoWriter | None = None

        # shutdown flag
        self._quit = threading.Event()

        self._key_thread = threading.Thread(target=self._key_listener, daemon=True)
        self._key_thread.start()

        self.get_logger().info("ImagePublisher ready — saves to: " + str(CAPTURES_DIR))

    # ── camera tick ──────────────────────────────────────────────────────────

    def tick(self):
        primary_bgr = grab_raw_frame(self.primary_pipe)
        wrist_bgr   = grab_raw_frame(self.wrist_pipe)
        stamp       = self.get_clock().now().to_msg()

        # publish downscaled frames to ROS2 topics (full-res goes to MP4 only)
        primary_pub = cv2.resize(primary_bgr, (PUB_WIDTH, PUB_HEIGHT),
                                 interpolation=cv2.INTER_AREA)
        wrist_pub   = cv2.resize(wrist_bgr,   (PUB_WIDTH, PUB_HEIGHT),
                                 interpolation=cv2.INTER_AREA)
        self.pub_primary.publish(
            numpy_to_image_msg(primary_pub, stamp, "camera_primary"))
        self.pub_wrist.publish(
            numpy_to_image_msg(wrist_pub,   stamp, "camera_wrist"))

        # write to MP4 if recording
        if self._recording:
            if self._writer_primary is not None:
                self._writer_primary.write(primary_bgr)
            if self._writer_wrist is not None:
                self._writer_wrist.write(wrist_bgr)

    # ── rosbag + video control ────────────────────────────────────────────────

    def _start_recording(self):
        if self._recording:
            self.get_logger().warn("Already recording — press S to stop first.")
            return

        self._demo_path  = next_demo_path()
        self._demo_path.mkdir(parents=True, exist_ok=True)
        self._demo_count += 1

        # start rosbag — log file sits next to the MP4s for diagnosis
        # bag_out  = self._demo_path / "bag"
        # log_file = open(self._demo_path / "rosbag.log", "w")
        # cmd = f"ros2 bag record -a -o {bag_out}"
        # self._bag_proc = subprocess.Popen(
        #     cmd, shell=True,
        #     stdout=log_file,
        #     stderr=log_file,
        #     preexec_fn=os.setsid,
        # )
        # self._bag_log = log_file

        # start MP4 writers — BGR format matches raw RealSense output
        self._writer_primary = cv2.VideoWriter(
            str(self._demo_path / "primary.mp4"),
            FOURCC, RATE_HZ,
            (STREAM_WIDTH, STREAM_HEIGHT),
        )
        self._writer_wrist = cv2.VideoWriter(
            str(self._demo_path / "wrist.mp4"),
            FOURCC, RATE_HZ,
            (STREAM_WIDTH, STREAM_HEIGHT),
        )

        self._recording = True

        self.get_logger().info(
            f"⏺  Recording demo {self._demo_count:03d} → {self._demo_path}"
        )

    def _stop_recording(self):
        if not self._recording:
            self.get_logger().warn("Not currently recording.")
            return

        # stop rosbag
        # os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGINT)
        # self._bag_proc.wait()
        # self._bag_proc = None
        # self._bag_log.close()
        # self._bag_log = None

        # finalize MP4 files
        if self._writer_primary is not None:
            self._writer_primary.release()
            self._writer_primary = None
        if self._writer_wrist is not None:
            self._writer_wrist.release()
            self._writer_wrist = None

        self._recording = False

        # self.get_logger().info(
        #     f"⏹  Saved  demo {self._demo_count:03d} → {self._demo_path}\n"
        #     f"         rosbag : {self._demo_path / 'bag'}\n"
        #     f"         primary: {self._demo_path / 'primary.mp4'}\n"
        #     f"         wrist  : {self._demo_path / 'wrist.mp4'}"
        # )
        self.get_logger().info(
            f"⏹  Saved  demo {self._demo_count:03d} → {self._demo_path}\n"
            f"         primary: {self._demo_path / 'primary.mp4'}\n"
            f"         wrist  : {self._demo_path / 'wrist.mp4'}"
        )
        self._demo_path = None

    # ── keypress listener ────────────────────────────────────────────────────

    def _key_listener(self):
        fd            = sys.stdin.fileno()
        original_term = termios.tcgetattr(fd)
        print("\nKeyboard ready — R=record  S=stop  Q=quit\n", flush=True)
        try:
            while not self._quit.is_set():
                key = get_keypress()
                if key == "r":
                    self._start_recording()
                elif key == "s":
                    self._stop_recording()
                elif key == "q":
                    print("\nQ pressed — shutting down.", flush=True)
                    if self._recording:
                        self._stop_recording()
                    self._quit.set()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, original_term)

    # ── cleanup ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        if self._recording:
            self._stop_recording()
        self._quit.set()
        self.get_logger().info("Stopping camera pipelines.")
        self.primary_pipe.stop()
        self.wrist_pipe.stop()
        super().destroy_node()
        try:
            os.system("stty sane")
        except Exception:
            pass


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ImagePublisher()
    try:
        while rclpy.ok() and not node._quit.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
