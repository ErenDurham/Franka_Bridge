"""
Written by: Makhtar N.

finetune_config.py  —  FR3 joint-position finetuning config for Octo

Usage (via pipeline.sh — recommended):
    ./pipeline.sh finetune

Usage (direct — run from /home/faro/octo_fr3/octo):
    python scripts/finetune.py \
        --config /home/faro/octo_fr3/src/finetune_config.py:full,language_conditioned \
        --config.pretrained_path hf://rail-berkeley/octo-base-1.5 \
        --config.save_dir /home/faro/octo_fr3/checkpoints \
        --config.dataset_kwargs.data_dir /home/faro/octo_fr3/TFDS_Data/tfds_output


    cd /home/faro/octo_fr3/octo && \
   export PYTHONPATH=/home/faro/octo_fr3/src:$PYTHONPATH && \
   python scripts/finetune.py \
       --config /home/faro/octo_fr3/src/finetune_config.py:full,language_conditioned \
       --config.pretrained_path hf://rail-berkeley/octo-base-1.5 \
       --config.save_dir /home/faro/octo_fr3/checkpoints \
       --config.dataset_kwargs.data_dir /home/faro/octo_fr3/TFDS_Data/tfds_output

Finetuning modes (first arg):
    full            — finetune everything (recommended for FR3, new action space)
    head_mlp_only   — freeze backbone, train MLP head only (faster, less data needed)
    head_only       — freeze everything except readout head (minimal compute)

Task modality (second arg):
    language_conditioned  — robot receives text instruction only
    image_conditioned     — robot receives goal image only
    multimodal            — randomly uses either (requires goal image in dataset)
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

    # ── dataset config ────────────────────────────────────────────────────────
    FINETUNING_KWARGS = {
        # Name must match the folder/class name of your TFDS builder
        "name": "fr3_demo_dataset",

        # Path where tfds build wrote the TFRecord shards — must be passed via CLI:
        #   --config.dataset_kwargs.data_dir /home/faro/octo_fr3/tfds_output
        "data_dir": placeholder(str),

        # Match the image keys defined in fr3_dataset_builder.py
        "image_obs_keys": {
            "primary": "image_primary",  # 3rd-person camera (256x256)
            "wrist":   "image_wrist",    # wrist camera (128x128)
        },

        # "proprio" is set by fr3_standardize_fn — 8-dim: 7 joints + gripper
        "proprio_obs_key": "proprio",

        # Language instruction key (set in fr3_standardize_fn under task)
        "language_key": "language_instruction",

        # Normalize actions — True means normalize that dimension.
        # 8 dimensions total: 7 joint positions (normalize) + 1 gripper (don't normalize)
        # Gripper is already in a meaningful [0,1] range so skip normalization.
        "action_normalization_mask": [True, True, True, True, True, True, True, False],

        # Action normalization strategy
        "action_proprio_normalization_type": "normal",

        # Your custom standardize function
        "standardize_fn": ModuleSpec.create(
            "fr3_standardize_fn:fr3_dataset_transform"
        ),
    }

    # ── frozen keys per mode ──────────────────────────────────────────────────
    if mode == "full":
        # Recommended for FR3: new action space needs full adaptation
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )

    
    max_steps  = FieldReference(50000)
    # window_size=1 fits in 8 GB VRAM (window_size=2 causes OOM due to 706-token sequence).
    # The 706→371 token reduction cuts attention memory from 2.2 GB to ~600 MB.
    # Increase to 2 if you move to a GPU with ≥16 GB VRAM.
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),

        # ── training hyperparameters ──────────────────────────────────────────
        # RTX 2000 Ada (8 GB VRAM): 8 is safe; increase to 16/32 if stable
        batch_size=8,
        shuffle_buffer_size=10000,
        num_steps=max_steps,

        # ── logging / saving ──────────────────────────────────────────────────
        log_interval=100,
        eval_interval=2000,   # evaluate on val set every N steps
        save_interval=2000,   # save checkpoint every N steps

        # ── early stopping (checked at every eval_interval) ───────────────────
        # Stops training once validation action MSE plateaus, so a 50k-step run
        # ends as soon as it stops improving. A checkpoint is saved at the
        # stopping step; the log also reports which saved step had the best
        # val MSE — deploy that one.
        early_stopping=dict(
            enabled=False,
            metric="mse",        # validation metric to track (from the action head)
            patience=3,          # consecutive evals without improvement → stop
            min_delta_pct=1.0,   # count as improvement only if ≥1% better than best
            min_steps=10000,     # never stop before this step (warmup + early noise)
        ),
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

        # ── optimizer ─────────────────────────────────────────────────────────
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=500,
                decay_steps=6250,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=8,
        ),

        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=4,   # 10 demos → 1 val episode; 4 batches is more than enough
        ),

        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=2,  # 1 val episode: 2 repeats is sufficient for stable metrics
            # trajs_for_viz disabled: plot_trajectory_actions in visualization_lib.py
            # hard-codes proprio slicing for EEF robots (indices 1:7 → 7-dim), which
            # clashes with our 8-dim joint-space action.  The 3D plot is meaningless
            # for joint-angle proprio anyway.
            trajs_for_viz=0,
            samples_per_state=8,
        ),
    )

    # ── task-specific data pipeline settings ──────────────────────────────────
    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5

    # ── trajectory transform ──────────────────────────────────────────────────
    traj_transform_kwargs = dict(
        window_size=window_size,

        # How many future actions Octo predicts per inference call.
        # At ~10Hz control, action_horizon=4 gives 0.4s of predicted motion
        # before the next Octo call. Increase if inference is slow.
        action_horizon=4,

        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
    )

    # ── frame (image) transform ───────────────────────────────────────────────
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        # No crop on wrist — framing is important for close-up manipulation
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),
            "wrist":   (128, 128),
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
            wrist=wrist_augment_kwargs,
        ),
    )

    config["frame_transform_threads"]  = 16
    config["traj_transform_kwargs"]    = traj_transform_kwargs
    config["frame_transform_kwargs"]   = frame_transform_kwargs

    # ── model config updates applied on top of pretrained octo-base-1.5 ──────────
    # octo-base-1.5 was pretrained with:
    #   observation_tokenizers: {primary: ImageTokenizer, wrist: ImageTokenizer}  (no proprio)
    #   heads.action: DiffusionActionHead(action_dim=7, action_horizon=4, ...)
    #
    # We add a proprio tokenizer and change action_dim from 7 → 8 (7 joints + gripper).
    # These are merged recursively into the pretrained config by finetune.py before
    # OctoModel.from_config() is called.  Mismatched params are randomly re-initialised
    # by merge_params() and then trained from scratch.
    config["update_config"] = dict(
        model=dict(
            observation_tokenizers=dict(
                # Add proprio tokenizer — pretrained model had none.
                # LowdimObsTokenizer concatenates observation["proprio"] (T, 8) and
                # feeds it to the transformer as continuous-valued tokens (discretize=False
                # matches the documented ALOHA finetuning example in octo/examples/02_*).
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
                # Replace pretrained action head (action_dim=7) with 8-dim version.
                # All other kwargs mirror the pretrained octo_pretrain_config.py values.
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
