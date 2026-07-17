"""
Written by: Makhtar N.

fr3_standardize_fn.py

Maps the FR3 demo dataset's observation/action keys into the names that
Octo's internal data pipeline expects.

Referenced in finetune_config.py as:
    "standardize_fn": ModuleSpec.create(
        "fr3_standardize_fn:fr3_dataset_transform"
    )

Octo's internal expected keys:
    observation:
        "image_primary"  → primary (3rd person) camera, (H, W, 3) uint8
        "image_wrist"    → wrist camera, (H, W, 3) uint8
        "proprio"        → proprioceptive state vector, (D,) float32
    action:              → action vector, (action_dim,) float32
    task:
        "language_instruction" → string task description
"""

import tensorflow as tf


def fr3_dataset_transform(trajectory: dict) -> dict:
    """
    Transform a single trajectory from FR3 dataset format to Octo format.

    Input trajectory keys (from fr3_dataset_builder.py):
        observation/image_primary   (T, 256, 256, 3) uint8
        observation/image_wrist     (T, 128, 128, 3) uint8
        observation/joint_positions (T, 7)           float32
        observation/gripper_state   (T, 1)           float32
        action                      (T, 8)           float32
        language_instruction        string

    Output adds/renames to match Octo's expected structure:
        observation/image_primary   — unchanged (Octo looks for this key)
        observation/image_wrist     — unchanged
        observation/proprio         (T, 8) float32  [joints + gripper concatenated]
        action                      (T, 8) float32  — unchanged
        task/language_instruction   — moved under task key
    """

    obs = trajectory["observation"]

    # ── proprio: concatenate joint positions + gripper state ──────────────────
    # Result: (T, 8) — 7 arm joints + 1 gripper width
    proprio = tf.concat([
        obs["joint_positions"],  # (T, 7)
        obs["gripper_state"],    # (T, 1)
    ], axis=-1)  # (T, 8)

    trajectory["observation"]["proprio"] = proprio

    # ── language instruction ───────────────────────────────────────────────────
    trajectory["language_instruction"] = trajectory["traj_metadata"]["language_instruction"]

    return trajectory


def fr3_dataset_transform_delta(trajectory: dict) -> dict:
    """
    Transform a single trajectory from FR3 dataset format to Octo format, and
    convert the action to a delta relative to the current joint positions.
    """
    trajectory = fr3_dataset_transform(trajectory)

    joint_deltas = (
        trajectory["action"][:, :7] - trajectory["observation"]["joint_positions"]
    )
    trajectory["action"] = tf.concat(
        [joint_deltas, trajectory["action"][:, 7:]], axis=-1
    )

    return trajectory
