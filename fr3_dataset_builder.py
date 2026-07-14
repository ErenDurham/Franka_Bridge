"""
Written by: Makhtar N.

fr3_dataset_builder.py  — ROOT COPY — DO NOT USE FOR BUILDING

This file is an outdated draft kept for reference only.

PROBLEMS with this version:
  1. Puts language_instruction INSIDE each step (inside the Dataset feature),
     but the TFDS data pipeline and fr3_standardize_fn.py expect it at the
     EPISODE level (outside steps). This produces the wrong dataset structure.
  2. Uses a dict (steps = {0: {...}, 1: {...}}) instead of a list for the steps
     sequence. TFDS Dataset features expect an iterable (list/generator).

The correct builder is:
    fr3_demo_dataset/fr3_dataset_builder.py

The dataset already built at tfds_output/fr3_demo_dataset/1.0.0/ was built
from the correct fr3_demo_dataset/ version — do not rebuild with this file.
"""

import os
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds

# ── dataset metadata ───────────────────────────────────────────────────────────

# Path to the HDF5 episode files produced by extract_bags.py.
# Override with --imports or set HDF5_EPISODE_DIR env var.
HDF5_EPISODE_DIR = os.environ.get("HDF5_EPISODE_DIR", "/path/to/hdf5_episodes")

TRAIN_SPLIT = 0.9  # 90% train, 10% val


class Fr3DemoDataset(tfds.core.GeneratorBasedBuilder):
    """
    RLDS dataset of FR3 arm demonstrations.

    Observation space:
        image_primary  : (256, 256, 3) uint8  — 3rd-person RGB
        image_wrist    : (128, 128, 3) uint8  — wrist RGB
        joint_positions: (7,)          float32 — measured arm joints
        gripper_state  : (1,)          float32 — measured gripper width (m)

    Action space (8-dim float32):
        [0:7] desired_joint_positions  — commanded arm joints
        [7]   gripper_action           — commanded gripper width (0-1)
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="FR3 robot arm demonstration dataset for Octo finetuning.",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_primary":   tfds.features.Image(
                                               shape=(256, 256, 3),
                                               dtype=tf.uint8,
                                               encoding_format="jpeg"),
                        "image_wrist":     tfds.features.Image(
                                               shape=(128, 128, 3),
                                               dtype=tf.uint8,
                                               encoding_format="jpeg"),
                        "joint_positions": tfds.features.Tensor(
                                               shape=(7,), dtype=tf.float32),
                        "gripper_state":   tfds.features.Tensor(
                                               shape=(1,), dtype=tf.float32),
                    }),
                    # 7 joint commands + 1 gripper command
                    "action":               tfds.features.Tensor(
                                                shape=(8,), dtype=tf.float32),
                    "language_instruction": tfds.features.Text(),
                    "is_first":             tf.bool,
                    "is_last":              tf.bool,
                    "is_terminal":          tf.bool,
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        episode_files = sorted([
            os.path.join(HDF5_EPISODE_DIR, f)
            for f in os.listdir(HDF5_EPISODE_DIR)
            if f.endswith(".hdf5")
        ])

        if not episode_files:
            raise FileNotFoundError(
                f"No .hdf5 files found in {HDF5_EPISODE_DIR}. "
                "Run extract_bags.py first, or set HDF5_EPISODE_DIR."
            )

        split_idx = int(len(episode_files) * TRAIN_SPLIT)
        train_files = episode_files[:split_idx]
        val_files   = episode_files[split_idx:]

        print(f"Dataset split — train: {len(train_files)}, val: {len(val_files)}")

        return {
            "train": self._generate_examples(train_files),
            "val":   self._generate_examples(val_files),
        }

    def _generate_examples(self, episode_files):
        for ep_idx, fpath in enumerate(episode_files):
            with h5py.File(fpath, "r") as f:
                # ── observations ──────────────────────────────────────────────
                joint_positions = f["observations/joint_positions"][:]  # (T, 7)
                gripper_state   = f["observations/gripper_state"][:]    # (T, 1)
                images_primary  = f["observations/images/primary"][:]   # (T, 256, 256, 3)
                images_wrist    = f["observations/images/wrist"][:]     # (T, 128, 128, 3)

                # ── actions ───────────────────────────────────────────────────
                actions = f["actions"][:]                               # (T, 8)

                # ── metadata ──────────────────────────────────────────────────
                lang    = str(f.attrs["language_instruction"])
                T       = len(actions)

            # build step-level dict
            steps = {}
            for t in range(T):
                steps[t] = {
                    "observation": {
                        "image_primary":   images_primary[t],
                        "image_wrist":     images_wrist[t],
                        "joint_positions": joint_positions[t].astype(np.float32),
                        "gripper_state":   gripper_state[t].astype(np.float32),
                    },
                    "action":               actions[t].astype(np.float32),
                    "language_instruction": lang,
                    "is_first":             t == 0,
                    "is_last":              t == T - 1,
                    "is_terminal":          t == T - 1,
                }

            yield ep_idx, {"steps": steps}
