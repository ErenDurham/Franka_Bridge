"""
Written by: Makhtar N.
fr3_dataset_builder.py
"""

import os
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds

HDF5_EPISODE_DIR = os.environ.get("HDF5_EPISODE_DIR", "/path/to/hdf5_episodes")
TRAIN_SPLIT = 0.9

class Fr3DemoDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_primary": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "image_wrist":   tfds.features.Image(shape=(128, 128, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "joint_positions": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                        "gripper_state":   tfds.features.Tensor(shape=(1,), dtype=np.float32),
                    }),
                    "action":    tfds.features.Tensor(shape=(8,), dtype=np.float32),
                    "is_first":  tfds.features.Scalar(dtype=np.bool_),
                    "is_last":   tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                }),
                "language_instruction": tfds.features.Text(),
            }),
        )

    def _split_generators(self, dl_manager):
        files = sorted([
            os.path.join(HDF5_EPISODE_DIR, f)
            for f in os.listdir(HDF5_EPISODE_DIR)
            if f.endswith(".hdf5")
        ])
        split = int(len(files) * TRAIN_SPLIT)
        return {
            "train": self._generate_examples(files[:split]),
            "val":   self._generate_examples(files[split:]),
        }

    def _generate_examples(self, files):
        for ep_idx, fpath in enumerate(files):
            with h5py.File(fpath, "r") as f:
                joint_positions = f["observations/joint_positions"][:]
                gripper_state   = f["observations/gripper_state"][:]
                images_primary  = f["observations/images/primary"][:]
                images_wrist    = f["observations/images/wrist"][:]
                actions         = f["actions"][:]
                lang            = str(f.attrs["language_instruction"])
                T               = len(actions)

            steps = []
            for t in range(T):
                steps.append({
                    "observation": {
                        "image_primary":   images_primary[t],
                        "image_wrist":     images_wrist[t],
                        "joint_positions": joint_positions[t].astype(np.float32),
                        "gripper_state":   gripper_state[t].astype(np.float32),
                    },
                    "action":      actions[t].astype(np.float32),
                    "is_first":    t == 0,
                    "is_last":     t == T - 1,
                    "is_terminal": t == T - 1,
                })

            yield ep_idx, {
                "steps": steps,
                "language_instruction": lang,
            }
