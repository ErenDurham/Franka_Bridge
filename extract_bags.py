"""
Written by: Makhtar N.

extract_bags.py

Converts ROS2 Humble rosbags from FR3 demonstrations into per-episode HDF5 files.

Usage:
    python extract_bags.py \
        --bag_dir /path/to/bags \
        --output_dir /path/to/hdf5_episodes \
        --language_instruction "pick up the cube"

Each bag = one demonstration episode → one HDF5 file.

Topics used:
    Observations:
        /camera/primary/image_raw                               -> images/primary
        /camera/wrist/image_raw                                 -> images/wrist
        /franka_robot_state_broadcaster/measured_joint_states   -> joint_positions (7,)
        /franka_gripper/joint_states                            -> gripper_state (1,)  [width]

    Actions (what was commanded):
        /gello/joint_states                                     -> joint_action (7,)
        /gripper/gripper_client/target_gripper_width_percent    -> gripper_action (1,)

    NOTE: /franka_robot_state_broadcaster/desired_joint_states is NOT used as
    the action source — with our joint impedance controller that topic holds a
    stale constant value for the entire bag. The GELLO leader-arm commands on
    /gello/joint_states are the true control signal.
"""

import os
import argparse
import numpy as np
import h5py
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge

# ── constants ──────────────────────────────────────────────────────────────────

TOPICS = {
    "image_primary":   "/camera/primary/image_raw",
    "image_wrist":     "/camera/wrist/image_raw",
    "measured_joints": "/franka_robot_state_broadcaster/measured_joint_states",
    "gello_joints":    "/gello/joint_states",
    "gripper_state":   "/franka_gripper/joint_states",
    "gripper_action":  "/gripper/gripper_client/target_gripper_width_percent",
}

IMAGE_SIZE_PRIMARY = (256, 256)
IMAGE_SIZE_WRIST   = (128, 128)
N_JOINTS           = 7  # FR3 arm DOF
Downsample_factor = 100
bridge = CvBridge()


# ── helpers ────────────────────────────────────────────────────────────────────

def nearest(timestamps: list, query: int) -> int:
    """Return index of closest timestamp to query."""
    ts = np.array(timestamps)
    return int(np.argmin(np.abs(ts - query)))


def at_or_before(timestamps: list, query: int) -> int:
    """Return index of the latest timestamp <= query (first message if none yet).
    """
    ts = np.array(timestamps)
    idx = np.searchsorted(ts, query, side="right") - 1
    return int(max(idx, 0))


def read_bag(bag_path: str) -> dict:
    """
    Read all messages from a rosbag and bucket them by topic.
    Returns dict: topic -> list of (timestamp_ns, msg)
    """
    reader = rosbag2_py.SequentialReader()
    storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_opts = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_opts, converter_opts)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    buckets = {v: [] for v in TOPICS.values()}

    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        if topic not in buckets:
            continue
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        buckets[topic].append((ts_ns, msg))

    return buckets


def extract_joints(msg) -> np.ndarray:
    """Extract 7 joint positions from a JointState message."""
    return np.array(msg.position[:N_JOINTS], dtype=np.float32)


def extract_gripper_state(msg) -> np.ndarray:
    """
    Extract gripper width from franka_gripper/joint_states.
    The two finger joints sum to total width — take first joint * 2.
    Returns scalar as (1,) array in metres.
    """
    width = float(msg.position[0]) * 2.0
    return np.array([width], dtype=np.float32)


def extract_gripper_action(msg) -> np.ndarray:
    """
    Extract gripper command from target_gripper_width_percent.
    Message is std_msgs/Float64 (value 0.0-1.0).
    We keep it as a normalized scalar.
    """
    return np.array([float(msg.data)], dtype=np.float32)


def decode_image(msg, size: tuple) -> np.ndarray:
    """Decode a sensor_msgs/Image to an RGB numpy array and resize."""
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    img = cv2.resize(img, (size[1], size[0]))  # cv2 uses (W, H)
    return img.astype(np.uint8)


def synchronize(buckets: dict) -> dict:
    """
    Use measured_joint_states as the reference clock.
    Snap all other streams to nearest timestamp.
    Returns arrays per stream, aligned to joint timestamps.
    """
    ref_stream = buckets[TOPICS["measured_joints"]]
    if not ref_stream:
        raise ValueError("No measured_joint_states messages found in bag.")

    ref_ts = [ts for ts, _ in ref_stream]
    T = len(ref_ts)

    def ts_list(topic):
        return [ts for ts, _ in buckets[topic]]

    def get_at(topic, i, match=nearest):
        return buckets[topic][match(ts_list(topic), ref_ts[i])][1]

    joint_positions = []
    gripper_states  = []
    gello_joints    = []
    gripper_actions = []
    images_primary  = []
    images_wrist    = []

    
    other_topics = [k for k in TOPICS if k != "measured_joints"]
    t_start = max(buckets[TOPICS[k]][0][0] for k in other_topics)
    first = int(np.searchsorted(np.array(ref_ts), t_start))
    if first >= T:
        raise ValueError("Streams only overlap after the last reference message.")

    # Downsampled reference indices (1 kHz -> 10 Hz)
    indices = list(range(first, T, Downsample_factor))

    for k, i in enumerate(indices):

        i_next = indices[k + 1] if k + 1 < len(indices) else i

        joint_positions.append(extract_joints(ref_stream[i][1]))
        gripper_states.append(extract_gripper_state(get_at(TOPICS["gripper_state"], i, at_or_before)))
        images_primary.append(decode_image(get_at(TOPICS["image_primary"], i, at_or_before), IMAGE_SIZE_PRIMARY))
        images_wrist.append(decode_image(get_at(TOPICS["image_wrist"], i, at_or_before), IMAGE_SIZE_WRIST))
        # Action labels: command nearest to t+1 (always after the observation).
        gello_joints.append(extract_joints(get_at(TOPICS["gello_joints"], i_next)))
        gripper_actions.append(extract_gripper_action(get_at(TOPICS["gripper_action"], i_next)))

    return {
        "joint_positions": np.stack(joint_positions),   # (T, 7)
        "gripper_state":   np.stack(gripper_states),    # (T, 1)
        "gello_joints":    np.stack(gello_joints),      # (T, 7)
        "gripper_actions": np.stack(gripper_actions),   # (T, 1)
        "images_primary":  np.stack(images_primary),    # (T, 256, 256, 3)
        "images_wrist":    np.stack(images_wrist),      # (T, 128, 128, 3)
    }


def compute_actions(synced: dict) -> np.ndarray:
    """
    Action at timestep t = GELLO joint command at t+1  (7,)
                         + gripper command at t+1       (1,)
    Combined: (T, 8)
    """
    joint_actions   = synced["gello_joints"]     # (T, 7)
    gripper_actions = synced["gripper_actions"]  # (T, 1)
    return np.concatenate([joint_actions, gripper_actions], axis=-1).astype(np.float32)  # (T, 8)


def save_episode(synced: dict, actions: np.ndarray, output_path: str, lang: str):
    with h5py.File(output_path, "w") as f:
        obs = f.create_group("observations")
        obs.create_dataset("joint_positions", data=synced["joint_positions"])  # (T, 7)
        obs.create_dataset("gripper_state",   data=synced["gripper_state"])    # (T, 1)

        imgs = obs.create_group("images")
        # Store as uint8 to keep file sizes manageable
        imgs.create_dataset("primary", data=synced["images_primary"],
                            dtype=np.uint8, compression="gzip", compression_opts=4)
        imgs.create_dataset("wrist",   data=synced["images_wrist"],
                            dtype=np.uint8, compression="gzip", compression_opts=4)

        f.create_dataset("actions", data=actions)  # (T, 8)
        f.attrs["language_instruction"] = lang
        f.attrs["num_steps"]            = len(actions)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_dir",  required=True,
                        help="Directory containing rosbag folders (one per episode)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write per-episode HDF5 files")
    parser.add_argument("--language_instruction", default="perform the task",
                        help="Language label for all episodes (or edit per-episode below)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bag_dirs = sorted([
        os.path.join(args.bag_dir, d)
        for d in os.listdir(args.bag_dir)
        if os.path.isdir(os.path.join(args.bag_dir, d))
    ])

    print(f"Found {len(bag_dirs)} bags.")

    for ep_idx, bag_path in enumerate(bag_dirs):
        print(f"[{ep_idx+1}/{len(bag_dirs)}] Processing {bag_path} ...")
        try:
            buckets = read_bag(bag_path)
            synced  = synchronize(buckets)
            actions = compute_actions(synced)

            out_path = os.path.join(args.output_dir, f"episode_{ep_idx:04d}.hdf5")
            save_episode(synced, actions, out_path, args.language_instruction)

            print(f"  → {synced['joint_positions'].shape[0]} steps saved to {out_path}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
