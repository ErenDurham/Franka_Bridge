"""
Written by: Makhtar N.

Usage (direct — run from the octo repo):
   export PYTHONPATH=<this-folder>:$PYTHONPATH && \
   python scripts/finetune.py \
       --config <this-folder>/finetune_config_oldCam.py:full,language_conditioned \
       --config.pretrained_path hf://rail-berkeley/octo-base-1.5 \
       --config.save_dir <checkpoint-dir> \
       --config.dataset_kwargs.data_dir <path-to>/tfds_output_oldcam

source /home/faro/miniconda3/etc/profile.d/conda.sh && conda activate octo_fr3
cd /home/faro/octo_fr3/octo
PYTHONPATH=/home/faro/octo_fr3/src python scripts/finetune.py \
  --config /home/faro/octo_fr3/src/finetune_config_oldCam.py:full,language_conditioned \
  --config.pretrained_path hf://rail-berkeley/octo-base-1.5 \
  --config.save_dir /home/faro/octo_fr3/checkpoints_oldCam \
  --config.dataset_kwargs.data_dir /home/faro/octo_fr3/src/tfds_output_oldcam

Finetuning modes (first arg):  full | head_mlp_only | head_only
Task modality (second arg):    language_conditioned | image_conditioned | multimodal
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.utils.spec import ModuleSpec


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # ── dataset ───────────────────────────────────────────────────────────────
    FINETUNING_KWARGS = {
        "name": "fr3_demo_dataset",
        "data_dir": placeholder(str),  # pass via --config.dataset_kwargs.data_dir
        "image_obs_keys": {"primary": "image_primary", "wrist": "image_wrist"},
        "proprio_obs_key": "proprio",           # 8-dim: 7 joints + gripper
        "language_key": "language_instruction",
        # 7 joints normalized, gripper (already in [0,1]) left as-is
        "action_normalization_mask": [True, True, True, True, True, True, True, False],
        "action_proprio_normalization_type": "normal",
        "standardize_fn": ModuleSpec.create("fr3_standardize_fn:fr3_dataset_transform"),
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )

    max_steps = FieldReference(6250)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=64,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=250,
        save_interval=250,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_fr3_finetune",
            group=placeholder(str),
            entity=placeholder(str),
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=500,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=4,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=2,
            trajs_for_viz=0,
            samples_per_state=8,
        ),
    )

    # ── task-conditioning pipeline settings ────────────────────────────────────
    if task == "image_conditioned":
        goal_relabeling_strategy, keep_image_prob = "uniform", 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy, keep_image_prob = None, 0.0
    elif task == "multimodal":
        goal_relabeling_strategy, keep_image_prob = "uniform", 0.5

    config["traj_transform_kwargs"] = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(keep_image_prob=keep_image_prob),
    )
    config["frame_transform_kwargs"] = dict(
        resize_size={"primary": (256, 256), "wrist": (128, 128)},
    )

    config["update_config"] = dict(
        model=dict(
            observation_tokenizers=dict(
                proprio=ModuleSpec.create(
                    LowdimObsTokenizer,
                    n_bins=256,
                    bin_type="normal",
                    low=-2.0,
                    high=2.0,
                    obs_keys=["proprio"],
                ),
            ),
            heads=dict(
                action=ModuleSpec.create(
                    DiffusionActionHead,
                    readout_key="readout_action",
                    use_map=False,
                    action_horizon=4,
                    action_dim=8,
                    n_diffusion_samples=4,
                    dropout_rate=0.0,
                ),
            ),
        ),
    )

    return ConfigDict(config)
