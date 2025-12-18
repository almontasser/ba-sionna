"""
Training Script for mmWave Beam Alignment Model

This script trains an end-to-end beam alignment model for mmWave communication systems
using adaptive sensing and feedback mechanisms. The model learns to select optimal
beam pairs between a Base Station (BS) and User Equipment (UE) through sequential
sensing steps guided by a recurrent neural network.

This implementation trains the full end-to-end beam alignment pipeline
(paper "C3" variant): UE RNN + BS FNN + learnable BS codebook.

Key paper-style parameters (Section IV): NTX=32, NRX=16, NCB=8,
batch_size=256, training SNR=10 dB, 2-layer GRU.

Note: model size is configurable; with the current default config the C3 model
is ~0.7M parameters (and can be reduced/increased via Config).

Usage:
    Basic training:
        $ python train.py
    
    Custom configuration:
        $ python train.py --epochs 200 --batch_size 512 --lr 0.0005
    
    Resume from checkpoint:
        $ python train.py --checkpoint_dir ./checkpoints/experiment_1
    
    Test mode (reduced dataset):
        $ python train.py --test_mode

Command-line Arguments:
    --epochs: Number of training epochs (default: Config.EPOCHS)
    --batch_size: Batch size for training (default: Config.BATCH_SIZE)
    --lr: Learning rate (default: Config.LEARNING_RATE)
    --checkpoint_dir: Directory to save checkpoints (default: Config.CHECKPOINT_DIR)
    --log_dir: Directory for TensorBoard logs (default: Config.LOG_DIR)
    --test_mode: Run in test mode with reduced dataset (1 epoch, 1000 samples)

Outputs:
    - Checkpoints saved to checkpoint_dir every 10 epochs and when validation improves
    - TensorBoard logs saved to log_dir for training visualization
    - Best model selected based on validation beamforming gain

Monitoring:
    View training progress with TensorBoard:
        $ tensorboard --logdir ./logs
"""

import tensorflow as tf
import os
from datetime import datetime
from tqdm import tqdm
import io
import contextlib
import threading
import queue
import random
import numpy as np

from device_setup import setup_device, print_device_info
from channel_model import SionnaScenarioChannelModel
from config import Config
from models.beam_alignment import BeamAlignmentModel
from checkpoint_utils import check_checkpoint_compatibility
from training.lr_schedule import ConstantLearningRate, ScaledSchedule, WarmupThenDecay, WarmupThenSchedule
from training.steps import sample_snr, train_step, validate


def create_model(config):
    """
    Create and initialize a BeamAlignmentModel from configuration.
    
    This function instantiates the complete end-to-end beam alignment system,
    including the BS controller, UE controller, and channel model components.
    
    Args:
        config: Configuration object (Config class) containing:
            - NTX: Number of BS transmit antennas
            - NRX: Number of UE receive antennas
            - NCB: BS codebook size
            - T: Number of sensing steps
            - RNN_HIDDEN_SIZE: UE RNN hidden state size
            - RNN_TYPE: UE RNN type ("GRU" or "LSTM")
            - NUM_FEEDBACK: Number of feedback values from UE to BS
    Returns:
        BeamAlignmentModel: Initialized beam alignment model ready for training
    
    Example:
        >>> from config import Config
        >>> model = create_model(Config)
        >>> # Model is now ready for training
        >>> results = model(batch_size=32, snr_db=10.0, training=True)
    """
    model = BeamAlignmentModel(
        num_tx_antennas=config.NTX,
        num_rx_antennas=config.NRX,
        codebook_size=config.NCB,
        num_sensing_steps=config.T,
        rnn_hidden_size=config.RNN_HIDDEN_SIZE,
        rnn_type=config.RNN_TYPE,
        num_feedback=config.NUM_FEEDBACK,
        start_beam_index=getattr(config, "START_BEAM_INDEX", 0),
        random_start=getattr(config, "RANDOM_START", True),
        carrier_frequency=config.CARRIER_FREQUENCY,
        scenarios=config.SCENARIOS,
        o2i_model=getattr(config, "O2I_MODEL", "low"),
        enable_pathloss=getattr(config, "ENABLE_PATHLOSS", False),
        enable_shadow_fading=getattr(config, "ENABLE_SHADOW_FADING", False),
        distance_range_m=getattr(config, "DISTANCE_RANGE_M", (10.0, 200.0)),
        ue_speed_range=getattr(config, "UE_SPEED_RANGE", (0.0, 30.0)),
        indoor_probability=getattr(config, "INDOOR_PROBABILITY", 0.0),
        ut_height_m=getattr(config, "UT_HEIGHT_M", 1.5),
        bs_height_umi_m=getattr(config, "BS_HEIGHT_UMI_M", 10.0),
        bs_height_uma_m=getattr(config, "BS_HEIGHT_UMA_M", 25.0),
        bs_height_rma_m=getattr(config, "BS_HEIGHT_RMA_M", 35.0),
    )

    return model


def _current_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
    lr = optimizer.learning_rate
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        return tf.cast(lr(optimizer.iterations), tf.float32)
    return tf.cast(lr, tf.float32)


_SCENARIO_CANONICAL = {
    "umi": "UMi",
    "uma": "UMa",
    "rma": "RMa",
    "UMi": "UMi",
    "UMa": "UMa",
    "RMa": "RMa",
}


def _canonical_scenario_name(name: str) -> str:
    key = str(name).strip()
    if key in _SCENARIO_CANONICAL:
        return _SCENARIO_CANONICAL[key]
    low = key.lower()
    if low in _SCENARIO_CANONICAL:
        return _SCENARIO_CANONICAL[low]
    raise ValueError(f"Unknown scenario '{name}'. Expected one of: UMi, UMa, RMa.")


def _parse_scenario_weights_spec(spec: str) -> dict[str, float]:
    """
    Parse: "UMi=0.6,UMa=0.3,RMa=0.1" -> {"UMi":0.6, "UMa":0.3, "RMa":0.1}
    """
    weights: dict[str, float] = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid --scenario_weights item '{part}'. Expected 'NAME=value', e.g. 'UMi=0.6'."
            )
        name, val = part.split("=", 1)
        scenario = _canonical_scenario_name(name)
        try:
            weights[scenario] = float(val)
        except ValueError as e:
            raise ValueError(
                f"Invalid weight for scenario '{scenario}': '{val}'. Expected a float."
            ) from e
    return weights


def _normalize_scenario_weights(
    scenarios: list[str],
    *,
    spec: str | None,
    w_umi: float | None,
    w_uma: float | None,
    w_rma: float | None,
) -> dict[str, float]:
    """
    Return normalized scenario weights over the active `scenarios`.

    - If no weights are provided: defaults to uniform over `scenarios`.
    - If weights are provided: missing scenarios default to 0 and then normalized.
    """
    scenarios = [_canonical_scenario_name(s) for s in scenarios]
    has_flag_weights = any(x is not None for x in (w_umi, w_uma, w_rma))
    if spec is not None and has_flag_weights:
        raise ValueError("Use either --scenario_weights or --w_umi/--w_uma/--w_rma, not both.")

    explicit = (spec is not None) or has_flag_weights
    raw: dict[str, float] = {}
    if spec is not None:
        raw = _parse_scenario_weights_spec(spec)
    elif has_flag_weights:
        raw = {
            "UMi": float(w_umi) if w_umi is not None else 0.0,
            "UMa": float(w_uma) if w_uma is not None else 0.0,
            "RMa": float(w_rma) if w_rma is not None else 0.0,
        }

    ignored = sorted(set(raw.keys()) - set(scenarios))
    if ignored:
        print(f"WARNING: Ignoring weights for inactive scenarios: {', '.join(ignored)}")

    if explicit:
        weights = {s: float(raw.get(s, 0.0)) for s in scenarios}
    else:
        weights = {s: 1.0 for s in scenarios}

    for s, w in weights.items():
        if w < 0.0:
            raise ValueError(f"Scenario weight for {s} must be non-negative, got {w}.")
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("Scenario weights must have at least one value > 0.")

    return {s: float(w) / total for s, w in weights.items()}


def _create_scenario_channel_model(config, scenario: str) -> SionnaScenarioChannelModel:
    scenario = _canonical_scenario_name(scenario)
    return SionnaScenarioChannelModel(
        num_tx_antennas=config.NTX,
        num_rx_antennas=config.NRX,
        carrier_frequency=config.CARRIER_FREQUENCY,
        scenarios=[scenario],
        o2i_model=getattr(config, "O2I_MODEL", "low"),
        enable_pathloss=getattr(config, "ENABLE_PATHLOSS", False),
        enable_shadow_fading=getattr(config, "ENABLE_SHADOW_FADING", False),
        distance_range_m=getattr(config, "DISTANCE_RANGE_M", (10.0, 200.0)),
        ue_speed_range=getattr(config, "UE_SPEED_RANGE", (0.0, 30.0)),
        indoor_probability=getattr(config, "INDOOR_PROBABILITY", 0.0),
        ut_height_m=getattr(config, "UT_HEIGHT_M", 1.5),
        bs_height_umi_m=getattr(config, "BS_HEIGHT_UMI_M", 10.0),
        bs_height_uma_m=getattr(config, "BS_HEIGHT_UMA_M", 25.0),
        bs_height_rma_m=getattr(config, "BS_HEIGHT_RMA_M", 35.0),
        fft_size=getattr(config, "RESOURCE_GRID_FFT_SIZE", 64),
        subcarrier_spacing=getattr(config, "RESOURCE_GRID_SUBCARRIER_SPACING", 120e3),
        narrowband_method=getattr(config, "NARROWBAND_METHOD", "center"),
        narrowband_subcarrier=getattr(config, "NARROWBAND_SUBCARRIER", None),
        generation_device=getattr(config, "CHANNEL_GENERATION_DEVICE", "auto"),
    )


def _resolve_channel_device(config) -> str:
    req = str(getattr(config, "CHANNEL_GENERATION_DEVICE", "auto")).lower()
    if req == "cpu":
        return "/CPU:0"
    if req == "gpu":
        if tf.config.list_physical_devices("GPU"):
            return "/GPU:0"
        print("WARNING: --channel_gen_device gpu requested but no GPU is visible; using CPU for channels.")
        return "/CPU:0"
    # auto
    return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"


def _build_per_scenario_batch_dataset(
    config,
    *,
    batch_size: int,
    num_time_samples: int,
    sampling_frequency: float,
    scenario_weights: dict[str, float],
) -> tuple[tf.data.Dataset, list[str]]:
    """
    Build an infinite dataset of (channels, scenario_id), where each element is a full batch
    drawn from exactly one TR 38.901 scenario (UMi/UMa/RMa) per the provided weights.
    """
    # Preserve the user's scenario order.
    scenarios = [_canonical_scenario_name(s) for s in getattr(config, "SCENARIOS", ["UMi", "UMa", "RMa"])]
    scenarios = [s for s in scenarios if s in scenario_weights]
    weights = [float(scenario_weights[s]) for s in scenarios]

    seed = int(getattr(config, "RANDOM_SEED", 0) or 0)
    gen_device = _resolve_channel_device(config)

    datasets: list[tf.data.Dataset] = []
    for sid, scenario in enumerate(scenarios):
        ch_model = _create_scenario_channel_model(config, scenario)

        ds = tf.data.Dataset.range(1).repeat()

        def _make_batch(_, ch_model=ch_model, sid=sid):
            with tf.device(gen_device):
                ch = ch_model.generate_channel(
                    int(batch_size),
                    num_time_samples=int(num_time_samples),
                    sampling_frequency=float(sampling_frequency),
                )
            return ch, tf.constant(sid, dtype=tf.int32)

        ds = ds.map(_make_batch, num_parallel_calls=1, deterministic=True)
        datasets.append(ds)

    mixed = tf.data.Dataset.sample_from_datasets(datasets, weights=weights, seed=seed)
    return mixed, scenarios


def train(config, checkpoint_dir=None, log_dir=None, *, run_name=None):
    """
    Main training loop.
    
    Args:
        config: Configuration object
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        run_name: Optional run name suffix for TensorBoard directory
    """
    print("=" * 80)
    print("BEAM ALIGNMENT TRAINING")
    print("=" * 80)
    
    # Setup device
    print("\nDevice Setup:")
    print_device_info()
    device_string, device_name = setup_device(verbose=False)
    
    # Enable mixed precision training for better GPU performance (~2x speedup on modern GPUs)
    if 'GPU' in device_string:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision training enabled (float16) for faster GPU training\n")
    
    # Print configuration
    print("\n")
    config.print_config()

    # Seed RNGs for reproducible LR ablations (TF + NumPy + Python).
    seed = int(getattr(config, "RANDOM_SEED", 0) or 0)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create directories
    if checkpoint_dir is None:
        checkpoint_dir = f"{config.CHECKPOINT_DIR}_C3_T{config.T}"
    if log_dir is None:
        log_dir = config.LOG_DIR
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = f"{str(run_name).strip()}_{timestamp}" if run_name else f"run_{timestamp}"
    run_dir = os.path.join(log_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(run_dir)

    # Log the full config once so runs are self-describing in TensorBoard.
    cfg_buf = io.StringIO()
    with contextlib.redirect_stdout(cfg_buf):
        config.print_config()
    with summary_writer.as_default():
        tf.summary.text("run/config", tf.constant(cfg_buf.getvalue()), step=0)

    # Scenario batch sampling weights (for multi-scenario training).
    scenario_weights = getattr(config, "SCENARIO_WEIGHTS", None)
    if scenario_weights is None:
        scenario_weights = _normalize_scenario_weights(
            list(getattr(config, "SCENARIOS", ["UMi", "UMa", "RMa"])),
            spec=None,
            w_umi=None,
            w_uma=None,
            w_rma=None,
        )
        config.SCENARIO_WEIGHTS = scenario_weights

    print("\nScenario batch sampling weights (normalized):")
    for s in ["UMi", "UMa", "RMa"]:
        if s in scenario_weights:
            print(f"  {s}: {scenario_weights[s]:.6f}")

    with summary_writer.as_default():
        tf.summary.text("run/scenario_weights", tf.constant(str(scenario_weights)), step=0)
        for s in ["UMi", "UMa", "RMa"]:
            if s in scenario_weights:
                tf.summary.scalar(f"run/scenario_weight_{s}", scenario_weights[s], step=0)

    print(f"\nTensorBoard run: {run_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    with tf.device(device_string):
        # Create model
        print(f"\nCreating model on {device_name}...")
        model = create_model(config)

        # Build model variables without generating TR 38.901 channels (can OOM on GPU
        # for large batch sizes and time-varying channels).
        print("Building model (no channel generation)...")
        model.build(None)

        # Setup checkpoint manager (model weights only).
        #
        # This is intentionally model-only: optimizer state is fragile across
        # architecture changes and can cause hard failures when restoring.
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=5
        )

        # Restore from checkpoint if available and compatible.
        start_epoch = 0
        if checkpoint_manager.latest_checkpoint:
            ckpt_path = checkpoint_manager.latest_checkpoint
            compat = check_checkpoint_compatibility(
                model,
                ckpt_path,
                require_all_trainable=True,
                require_ue_rnn_kernel=True,
            )
            if compat.ok:
                try:
                    checkpoint.restore(ckpt_path).expect_partial()
                    print(f"✓ Restored model weights from {ckpt_path}")
                except (ValueError, tf.errors.InvalidArgumentError) as e:
                    print(
                        "WARNING: Found a checkpoint but restore failed due to incompatibility.\n"
                        "Starting training from scratch.\n"
                        f"Restore error: {e}"
                    )
                    # Avoid overwriting incompatible checkpoints by switching
                    # to a fresh directory.
                    fresh_dir = f"{checkpoint_dir}_fresh_{timestamp}"
                    print(f"Using fresh checkpoint dir: {fresh_dir}")
                    os.makedirs(fresh_dir, exist_ok=True)
                    checkpoint_dir = fresh_dir
                    checkpoint_manager = tf.train.CheckpointManager(
                        checkpoint, checkpoint_dir, max_to_keep=5
                    )
            else:
                print(
                    "WARNING: Found a checkpoint but it does not match the current model.\n"
                    "Starting training from scratch.\n"
                    "Details:\n"
                    f"{compat.summary()}\n"
                    "Tip: use a fresh --checkpoint_dir (or delete old checkpoints) after changing architecture."
                )
                fresh_dir = f"{checkpoint_dir}_fresh_{timestamp}"
                print(f"Using fresh checkpoint dir: {fresh_dir}")
                os.makedirs(fresh_dir, exist_ok=True)
                checkpoint_dir = fresh_dir
                checkpoint_manager = tf.train.CheckpointManager(
                    checkpoint, checkpoint_dir, max_to_keep=5
                )
        else:
            print("Starting training from scratch")

        # Create optimizer with learning-rate schedule
        steps_per_epoch = max(1, config.NUM_TRAIN_SAMPLES // config.BATCH_SIZE)
        base_lr = float(config.LEARNING_RATE)
        lr_scale = float(getattr(config, "LR_SCALE", 1.0))
        warmup_epochs = int(getattr(config, "LR_WARMUP_EPOCHS", 0) or 0)
        warmup_steps = int(warmup_epochs * steps_per_epoch)

        lr_schedule_name = str(getattr(config, "LR_SCHEDULE", "warmup_then_decay")).lower()
        if lr_schedule_name in {"warmup_then_decay", "warmup_decay", "exp_decay"}:
            decay_steps = max(1, int(config.LEARNING_RATE_DECAY_STEPS * steps_per_epoch))
            base_schedule = WarmupThenDecay(
                base_lr=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                decay_rate=config.LEARNING_RATE_DECAY,
            )
        elif lr_schedule_name in {"cosine_restarts", "cosine_restart", "cosine"}:
            first_decay_epochs = float(getattr(config, "LR_COSINE_FIRST_DECAY_EPOCHS", 10))
            first_decay_steps = max(1, int(first_decay_epochs * steps_per_epoch))
            cosine = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=base_lr,
                first_decay_steps=first_decay_steps,
                t_mul=float(getattr(config, "LR_COSINE_T_MUL", 2.0)),
                m_mul=float(getattr(config, "LR_COSINE_M_MUL", 1.0)),
                alpha=float(getattr(config, "LR_COSINE_ALPHA", 0.0)),
            )
            base_schedule = WarmupThenSchedule(base_lr=base_lr, warmup_steps=warmup_steps, schedule=cosine)
        elif lr_schedule_name in {"constant", "const"}:
            const = ConstantLearningRate(base_lr)
            base_schedule = WarmupThenSchedule(base_lr=base_lr, warmup_steps=warmup_steps, schedule=const)
        else:
            raise ValueError(
                f"Unknown LR_SCHEDULE={lr_schedule_name!r}. "
                "Expected one of: warmup_then_decay, cosine_restarts, constant."
            )

        lr_schedule = ScaledSchedule(base_schedule, initial_scale=lr_scale)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        print("\nOptimizer/LR:")
        print(f"  LR schedule: {lr_schedule_name}")
        print(f"  Base LR: {base_lr:g}")
        print(f"  LR scale: {lr_scale:g}")
        if warmup_steps > 0:
            print(f"  Warmup: {warmup_epochs} epochs ({warmup_steps} steps)")
        if lr_schedule_name in {"cosine_restarts", "cosine_restart", "cosine"}:
            print(f"  Cosine first period: {first_decay_steps} steps (~{first_decay_epochs:g} epochs)")
        
        # Training loop
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        steps_per_epoch = max(1, config.NUM_TRAIN_SAMPLES // config.BATCH_SIZE)
        val_batches = max(1, config.NUM_VAL_SAMPLES // config.BATCH_SIZE)
        
        global_step = start_epoch * steps_per_epoch
        best_val_bf_gain = -float('inf')
        num_backoffs = 0
        skipped_steps_in_row = 0

        # Channel time parameters (used for per-scenario batch datasets).
        num_time_samples = 1
        sampling_frequency = 1.0
        if getattr(config, "MOBILITY_ENABLE", False):
            nts = getattr(config, "MOBILITY_NUM_TIME_SAMPLES", None)
            num_time_samples = int(nts) if nts is not None else int(config.T + 1)
            sampling_frequency = float(getattr(config, "MOBILITY_SAMPLING_FREQUENCY_HZ", 1.0))

        # Pre-generate a fixed validation set so epoch-to-epoch comparisons
        # reflect learning, not resampling variance (mirrors Fig. 4 eval style).
        val_channels_batches = None
        val_start_idx_batches = None
        if bool(getattr(config, "VAL_USE_FIXED_CHANNELS", False)):
            if not bool(getattr(config, "TRAIN_CHANNELS_OUTSIDE_GRAPH", False)):
                print(
                    "WARNING: VAL_USE_FIXED_CHANNELS=1 requires TRAIN_CHANNELS_OUTSIDE_GRAPH=1. "
                    "Disabling fixed validation channels."
                )
            else:
                val_batches_eff = max(2, min(int(val_batches), 3))
                val_batch_size = int(config.BATCH_SIZE) * 2
                print(
                    f"\nPre-generating fixed validation set: {val_batches_eff} batches "
                    f"(batch_size={val_batch_size})..."
                )
                val_channels_batches = []
                for _ in tqdm(range(val_batches_eff), desc="Val channels"):
                    ch, _, _ = model.generate_channels(val_batch_size)
                    val_channels_batches.append(ch)

                if bool(getattr(config, "VAL_USE_FIXED_START_IDX", False)):
                    with tf.device("/CPU:0"):
                        val_start_idx_batches = [
                            tf.random.uniform(
                                [val_batch_size],
                                minval=0,
                                maxval=int(config.NCB),
                                dtype=tf.int32,
                            )
                            for _ in range(val_batches_eff)
                        ]
                print("✓ Fixed validation set ready")

        # Per-scenario tf.data pipeline: one scenario per batch, sampled by weights.
        train_scenario_ds = None
        train_scenario_iter = None
        train_scenarios = None
        use_scenario_ds = len(getattr(config, "SCENARIOS", [])) > 1
        if use_scenario_ds:
            cache_size = int(getattr(config, "CHANNEL_CACHE_SIZE", 0) or 0)
            if cache_size > 0:
                print("Note: CHANNEL_CACHE_SIZE is ignored when using per-scenario tf.data sampling.")
            train_scenario_ds, train_scenarios = _build_per_scenario_batch_dataset(
                config,
                batch_size=int(config.BATCH_SIZE),
                num_time_samples=int(num_time_samples),
                sampling_frequency=float(sampling_frequency),
                scenario_weights=scenario_weights,
            )
            train_scenario_iter = iter(train_scenario_ds)
            print("\nTraining data: per-scenario batches via tf.data.sample_from_datasets")
            for s in train_scenarios:
                print(f"  {s}: {scenario_weights.get(s, 0.0):.6f}")

        def _backoff_lr(reason: str) -> None:
            nonlocal num_backoffs, skipped_steps_in_row
            max_backoffs = int(getattr(config, "LR_MAX_BACKOFFS", 3))
            if num_backoffs >= max_backoffs:
                raise RuntimeError(
                    f"Training diverged ({reason}) and exceeded LR_MAX_BACKOFFS={max_backoffs}. "
                    "Lower --lr/--lr_scale and retry."
                )
            factor = float(getattr(config, "LR_BACKOFF_FACTOR", 0.5))
            lr_schedule.multiply_scale(factor)
            num_backoffs += 1
            skipped_steps_in_row = 0
            try:
                new_lr = float(_current_learning_rate(optimizer).numpy())
                new_scale = float(lr_schedule.scale.numpy())
                print(
                    f"WARNING: LR backoff ({reason}). "
                    f"lr_scale={new_scale:g}, lr={new_lr:.6g} (backoffs={num_backoffs}/{max_backoffs})"
                )
            except Exception:
                print(f"WARNING: LR backoff ({reason}).")
        
        for epoch in range(start_epoch, config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
            print("-" * 80)
            try:
                lr_now = float(_current_learning_rate(optimizer).numpy())
                iter_now = int(optimizer.iterations.numpy())
                print(f"LR @ step {iter_now}: {lr_now:.6g}")
            except Exception:
                pass
            
            # Training
            epoch_loss = 0.0
            epoch_bf_gain = 0.0

            if use_scenario_ds and train_scenario_iter is not None:
                # One scenario per batch, chosen by the configured weights.
                pbar = tqdm(range(steps_per_epoch), desc="Training")
                for step in pbar:
                    channels, scenario_id = next(train_scenario_iter)
                    snr_db = sample_snr(config)

                    loss, bf_gain_db, grad_norm, update_skipped = train_step(
                        model,
                        optimizer,
                        config.BATCH_SIZE,
                        snr_db,
                        channels=channels,
                    )

                    loss_val = float(loss.numpy())
                    bf_gain_val = float(bf_gain_db.numpy())
                    grad_norm_val = float(grad_norm.numpy())
                    skipped_val = int(update_skipped.numpy())

                    epoch_loss += loss_val
                    epoch_bf_gain += bf_gain_val
                    global_step += 1

                    if skipped_val:
                        skipped_steps_in_row += 1
                        max_skips = int(getattr(config, "DIVERGENCE_MAX_SKIPPED_STEPS", 20))
                        if skipped_steps_in_row >= max_skips:
                            _backoff_lr(f"{skipped_steps_in_row} non-finite steps in a row")
                    else:
                        skipped_steps_in_row = 0

                    gain_norm = -loss
                    gain_norm_val = -loss_val
                    pbar_dict = {
                        'loss': f'{loss_val:.4f}',
                        'gain_norm': f'{gain_norm_val:.4f}',
                        'BF_gain': f'{bf_gain_val:.2f} dB',
                        'grad_norm': f'{grad_norm_val:.3f}',
                        'skipped': str(skipped_val),
                    }
                    pbar.set_postfix(pbar_dict)

                    if step % 10 == 0:
                        with summary_writer.as_default():
                            tf.summary.scalar("train/loss", loss, step=global_step)
                            tf.summary.scalar("train/gain_norm", gain_norm, step=global_step)
                            tf.summary.scalar("train/bf_gain_db", bf_gain_db, step=global_step)
                            tf.summary.scalar("train/gradient_norm", grad_norm, step=global_step)
                            tf.summary.scalar(
                                "train/update_skipped", tf.cast(update_skipped, tf.float32), step=global_step
                            )
                            tf.summary.scalar(
                                "train/learning_rate", _current_learning_rate(optimizer), step=global_step
                            )
            elif getattr(config, "TRAIN_CHANNELS_OUTSIDE_GRAPH", False):
                # Legacy per-batch training: generate channels in Python (mixed scenarios within batch).
                cache_size = getattr(config, "CHANNEL_CACHE_SIZE", 0)

                if cache_size > 0:
                    # Pre-generate channel cache at epoch start
                    if epoch == start_epoch or not hasattr(train, '_channel_cache'):
                        print(f"Pre-generating {cache_size} channel batches for cache...")
                        train._channel_cache = []
                        for _ in tqdm(range(cache_size), desc="Caching channels"):
                            channels, _, _ = model.generate_channels(config.BATCH_SIZE)
                            train._channel_cache.append(channels)
                        print(f"✓ Channel cache ready ({cache_size} batches)")

                    # Training using cached channels (random sampling)
                    pbar = tqdm(range(steps_per_epoch), desc="Training")
                    for step in pbar:
                        cache_idx = tf.random.uniform([], 0, cache_size, dtype=tf.int32).numpy()
                        channels = train._channel_cache[cache_idx]
                        snr_db = sample_snr(config)

                        loss, bf_gain_db, grad_norm, update_skipped = train_step(
                            model,
                            optimizer,
                            config.BATCH_SIZE,
                            snr_db,
                            channels=channels,
                        )

                        loss_val = float(loss.numpy())
                        bf_gain_val = float(bf_gain_db.numpy())
                        grad_norm_val = float(grad_norm.numpy())
                        skipped_val = int(update_skipped.numpy())

                        epoch_loss += loss_val
                        epoch_bf_gain += bf_gain_val
                        global_step += 1

                        if skipped_val:
                            skipped_steps_in_row += 1
                            max_skips = int(getattr(config, "DIVERGENCE_MAX_SKIPPED_STEPS", 20))
                            if skipped_steps_in_row >= max_skips:
                                _backoff_lr(f"{skipped_steps_in_row} non-finite steps in a row")
                        else:
                            skipped_steps_in_row = 0

                        gain_norm = -loss
                        gain_norm_val = -loss_val
                        pbar_dict = {
                            'loss': f'{loss_val:.4f}',
                            'gain_norm': f'{gain_norm_val:.4f}',
                            'BF_gain': f'{bf_gain_val:.2f} dB',
                            'grad_norm': f'{grad_norm_val:.3f}',
                            'skipped': str(skipped_val),
                        }
                        pbar.set_postfix(pbar_dict)

                        if step % 10 == 0:
                            with summary_writer.as_default():
                                tf.summary.scalar("train/loss", loss, step=global_step)
                                tf.summary.scalar("train/gain_norm", gain_norm, step=global_step)
                                tf.summary.scalar("train/bf_gain_db", bf_gain_db, step=global_step)
                                tf.summary.scalar("train/gradient_norm", grad_norm, step=global_step)
                                tf.summary.scalar(
                                    "train/update_skipped", tf.cast(update_skipped, tf.float32), step=global_step
                                )
                                tf.summary.scalar(
                                    "train/learning_rate", _current_learning_rate(optimizer), step=global_step
                                )
                else:
                    # No caching - generate fresh channels each iteration (slow but max diversity)
                    pbar = tqdm(range(steps_per_epoch), desc="Training")
                    for step in pbar:
                        channels, _, _ = model.generate_channels(config.BATCH_SIZE)
                        snr_db = sample_snr(config)

                        loss, bf_gain_db, grad_norm, update_skipped = train_step(
                            model,
                            optimizer,
                            config.BATCH_SIZE,
                            snr_db,
                            channels=channels,
                        )

                        loss_val = float(loss.numpy())
                        bf_gain_val = float(bf_gain_db.numpy())
                        grad_norm_val = float(grad_norm.numpy())
                        skipped_val = int(update_skipped.numpy())

                        epoch_loss += loss_val
                        epoch_bf_gain += bf_gain_val
                        global_step += 1

                        if skipped_val:
                            skipped_steps_in_row += 1
                            max_skips = int(getattr(config, "DIVERGENCE_MAX_SKIPPED_STEPS", 20))
                            if skipped_steps_in_row >= max_skips:
                                _backoff_lr(f"{skipped_steps_in_row} non-finite steps in a row")
                        else:
                            skipped_steps_in_row = 0

                        gain_norm = -loss
                        gain_norm_val = -loss_val
                        pbar_dict = {
                            'loss': f'{loss_val:.4f}',
                            'gain_norm': f'{gain_norm_val:.4f}',
                            'BF_gain': f'{bf_gain_val:.2f} dB',
                            'grad_norm': f'{grad_norm_val:.3f}',
                            'skipped': str(skipped_val),
                        }
                        pbar.set_postfix(pbar_dict)

                        if step % 10 == 0:
                            with summary_writer.as_default():
                                tf.summary.scalar("train/loss", loss, step=global_step)
                                tf.summary.scalar("train/gain_norm", gain_norm, step=global_step)
                                tf.summary.scalar("train/bf_gain_db", bf_gain_db, step=global_step)
                                tf.summary.scalar("train/gradient_norm", grad_norm, step=global_step)
                                tf.summary.scalar(
                                    "train/update_skipped", tf.cast(update_skipped, tf.float32), step=global_step
                                )
                                tf.summary.scalar(
                                    "train/learning_rate", _current_learning_rate(optimizer), step=global_step
                                )
            else:
                # Fallback: generate channels inside the training step (legacy graph mode).
                pbar = tqdm(range(steps_per_epoch), desc="Training")
                for step in pbar:
                    snr_db = sample_snr(config)
                    loss, bf_gain_db, grad_norm, update_skipped = train_step(
                        model,
                        optimizer,
                        config.BATCH_SIZE,
                        snr_db,
                        channels=None,
                    )

                    loss_val = float(loss.numpy())
                    bf_gain_val = float(bf_gain_db.numpy())
                    grad_norm_val = float(grad_norm.numpy())
                    skipped_val = int(update_skipped.numpy())

                    epoch_loss += loss_val
                    epoch_bf_gain += bf_gain_val
                    global_step += 1

                    if skipped_val:
                        skipped_steps_in_row += 1
                        max_skips = int(getattr(config, "DIVERGENCE_MAX_SKIPPED_STEPS", 20))
                        if skipped_steps_in_row >= max_skips:
                            _backoff_lr(f"{skipped_steps_in_row} non-finite steps in a row")
                    else:
                        skipped_steps_in_row = 0

                    gain_norm = -loss
                    gain_norm_val = -loss_val
                    pbar_dict = {
                        'loss': f'{loss_val:.4f}',
                        'gain_norm': f'{gain_norm_val:.4f}',
                        'BF_gain': f'{bf_gain_val:.2f} dB',
                        'grad_norm': f'{grad_norm_val:.3f}',
                        'skipped': str(skipped_val),
                    }
                    pbar.set_postfix(pbar_dict)

                    if step % 10 == 0:
                        with summary_writer.as_default():
                            tf.summary.scalar("train/loss", loss, step=global_step)
                            tf.summary.scalar("train/gain_norm", gain_norm, step=global_step)
                            tf.summary.scalar("train/bf_gain_db", bf_gain_db, step=global_step)
                            tf.summary.scalar("train/gradient_norm", grad_norm, step=global_step)
                            tf.summary.scalar(
                                "train/update_skipped", tf.cast(update_skipped, tf.float32), step=global_step
                            )
                            tf.summary.scalar(
                                "train/learning_rate", _current_learning_rate(optimizer), step=global_step
                            )
            
            # Epoch statistics
            avg_loss = epoch_loss / steps_per_epoch
            avg_bf_gain = epoch_bf_gain / steps_per_epoch
            
            print(f"\nTraining - Loss: {avg_loss:.4f}, BF Gain: {avg_bf_gain:.2f} dB")
            
            # Validation
            print("Validating...")
            val_metrics = validate(
                model, val_batches, config.BATCH_SIZE, 
                config.SNR_TRAIN, config.SNR_TARGET,
                channels_batches=val_channels_batches,
                start_idx_batches=val_start_idx_batches,
            )
            
            print(f"Validation - Loss: {val_metrics['val_loss']:.4f}")
            print(f"             BF Gain: {val_metrics['mean_bf_gain_db']:.2f} ± {val_metrics['std_bf_gain_db']:.2f} dB")
            print(f"             Satisfaction Prob: {val_metrics['satisfaction_prob']:.3f}")
            
            # Log to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar("val/loss", val_metrics["val_loss"], step=global_step)
                tf.summary.scalar("val/bf_gain_db", val_metrics["mean_bf_gain_db"], step=global_step)
                tf.summary.scalar(
                    "val/satisfaction_prob", val_metrics["satisfaction_prob"], step=global_step
                )
            
            # Save checkpoint
            if val_metrics['mean_bf_gain_db'] > best_val_bf_gain:
                best_val_bf_gain = val_metrics['mean_bf_gain_db']
                save_path = checkpoint_manager.save()
                print(f"✓ New best model! Saved checkpoint: {save_path}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                save_path = checkpoint_manager.save()
                print(f"Saved periodic checkpoint: {save_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation BF gain: {best_val_bf_gain:.2f} dB")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {log_dir}")
    print(f"  (this run: {run_dir})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train beam alignment model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument(
        '--lr_schedule',
        type=str,
        default=None,
        choices=["warmup_then_decay", "cosine_restarts", "constant"],
        help='Learning-rate schedule (overrides Config.LR_SCHEDULE)',
    )
    parser.add_argument(
        '--lr_scale',
        type=float,
        default=None,
        help='Global LR multiplier (overrides Config.LR_SCALE)',
    )
    parser.add_argument(
        '--cosine_first_decay_epochs',
        type=float,
        default=None,
        help='Cosine restarts: first decay period in epochs (overrides Config.LR_COSINE_FIRST_DECAY_EPOCHS)',
    )
    parser.add_argument(
        '--cosine_t_mul',
        type=float,
        default=None,
        help='Cosine restarts: cycle length multiplier t_mul (overrides Config.LR_COSINE_T_MUL)',
    )
    parser.add_argument(
        '--cosine_m_mul',
        type=float,
        default=None,
        help='Cosine restarts: amplitude multiplier m_mul (overrides Config.LR_COSINE_M_MUL)',
    )
    parser.add_argument(
        '--cosine_alpha',
        type=float,
        default=None,
        help='Cosine restarts: minimum LR fraction alpha (overrides Config.LR_COSINE_ALPHA)',
    )
    parser.add_argument('--num_sensing_steps', '-T', type=int, default=None, 
                       help='Number of sensing steps (T). Default: Config.T (16)')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for logs/checkpoints')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides Config.RANDOM_SEED)')
    parser.add_argument(
        '--require_gpu',
        action='store_true',
        help='Fail fast if no GPU is visible to TensorFlow',
    )
    parser.add_argument(
        '--channel_gen_device',
        type=str,
        default=None,
        choices=['auto', 'cpu', 'gpu'],
        help='Channel generation device placement (overrides Config.CHANNEL_GENERATION_DEVICE)',
    )
    parser.add_argument(
        '--train_channels_outside_graph',
        type=int,
        default=None,
        choices=[0, 1],
        help='If 1, generate channels outside @tf.function (better GPU placement); overrides Config.TRAIN_CHANNELS_OUTSIDE_GRAPH',
    )
    parser.add_argument(
        '--scenarios',
        type=str,
        default=None,
        help='Comma-separated list of scenarios to use (e.g., "UMi,UMa,RMa")',
    )
    parser.add_argument(
        '--scenario_weights',
        type=str,
        default=None,
        help='Per-scenario batch sampling weights, e.g. "UMi=0.6,UMa=0.3,RMa=0.1".',
    )
    parser.add_argument('--w_umi', type=float, default=None, help='Weight for UMi (batch sampling)')
    parser.add_argument('--w_uma', type=float, default=None, help='Weight for UMa (batch sampling)')
    parser.add_argument('--w_rma', type=float, default=None, help='Weight for RMa (batch sampling)')
    parser.add_argument('--target_snr', type=float, default=None,
                       help='Target SNR (dB) for satisfaction probability metrics')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                       help='Number of warm-up epochs with linear LR ramp (0 = no warm-up)')
    parser.add_argument('--snr_train_range', type=str, default=None,
                       help='Training SNR range "low,high" in dB (e.g., "0,10")')
    parser.add_argument(
        '--xla_jit',
        type=int,
        default=None,
        choices=[0, 1],
        help='Enable XLA JIT compilation (1=on, 0=off). Default: on for GPU. Disable if XLA causes errors.',
    )
    parser.add_argument(
        '--channel_cache_size',
        type=int,
        default=None,
        help='Number of channel batches to pre-generate and cache (0=disabled). Default: 100.',
    )
    parser.add_argument(
        '--val_fixed_channels',
        type=int,
        default=None,
        choices=[0, 1],
        help='If 1, use fixed validation channels across epochs; overrides Config.VAL_USE_FIXED_CHANNELS',
    )
    parser.add_argument(
        '--val_fixed_start_idx',
        type=int,
        default=None,
        choices=[0, 1],
        help='If 1, use fixed validation sweep start indices; overrides Config.VAL_USE_FIXED_START_IDX',
    )
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (1 epoch)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs is not None:
        Config.EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
    if args.lr_schedule is not None:
        Config.LR_SCHEDULE = args.lr_schedule
    if args.lr_scale is not None:
        Config.LR_SCALE = float(args.lr_scale)
    if args.cosine_first_decay_epochs is not None:
        Config.LR_COSINE_FIRST_DECAY_EPOCHS = float(args.cosine_first_decay_epochs)
    if args.cosine_t_mul is not None:
        Config.LR_COSINE_T_MUL = float(args.cosine_t_mul)
    if args.cosine_m_mul is not None:
        Config.LR_COSINE_M_MUL = float(args.cosine_m_mul)
    if args.cosine_alpha is not None:
        Config.LR_COSINE_ALPHA = float(args.cosine_alpha)
    if args.lr_warmup_epochs is not None:
        Config.LR_WARMUP_EPOCHS = args.lr_warmup_epochs
    if args.num_sensing_steps is not None:
        Config.T = args.num_sensing_steps
    if args.scenarios is not None:
        Config.SCENARIOS = [s.strip() for s in args.scenarios.split(',') if s.strip()]
    if args.target_snr is not None:
        Config.SNR_TARGET = args.target_snr
    if args.channel_gen_device is not None:
        Config.CHANNEL_GENERATION_DEVICE = args.channel_gen_device
    if args.train_channels_outside_graph is not None:
        Config.TRAIN_CHANNELS_OUTSIDE_GRAPH = bool(int(args.train_channels_outside_graph))
    if args.snr_train_range is not None:
        try:
            low, high = args.snr_train_range.split(',')
            Config.SNR_TRAIN_RANGE = (float(low), float(high))
        except Exception as e:
            raise ValueError(f"Invalid --snr_train_range '{args.snr_train_range}'. Expected format: low,high") from e
    if args.xla_jit is not None:
        Config.XLA_JIT_COMPILE = bool(args.xla_jit)
    if args.channel_cache_size is not None:
        Config.CHANNEL_CACHE_SIZE = args.channel_cache_size
    if args.val_fixed_channels is not None:
        Config.VAL_USE_FIXED_CHANNELS = bool(int(args.val_fixed_channels))
    if args.val_fixed_start_idx is not None:
        Config.VAL_USE_FIXED_START_IDX = bool(int(args.val_fixed_start_idx))
    if args.seed is not None:
        Config.RANDOM_SEED = int(args.seed)

    # Scenario weights (batch-level sampling)
    Config.SCENARIO_WEIGHTS = _normalize_scenario_weights(
        list(getattr(Config, "SCENARIOS", ["UMi", "UMa", "RMa"])),
        spec=args.scenario_weights,
        w_umi=args.w_umi,
        w_uma=args.w_uma,
        w_rma=args.w_rma,
    )
    
    if args.test_mode:
        Config.EPOCHS = 1
        Config.NUM_TRAIN_SAMPLES = 1000
        Config.NUM_VAL_SAMPLES = 200
        print("Running in TEST MODE (reduced dataset)")
    
    # Set checkpoint directory (C3-only) if not explicitly provided
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        suffix = f"_{args.run_name}" if args.run_name else ""
        checkpoint_dir = f"./checkpoints_C3_T{Config.T}{suffix}"
    
    if args.require_gpu:
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError(
                "No GPU visible to TensorFlow, but --require_gpu was set. "
                "Check your CUDA/TensorFlow install and CUDA_VISIBLE_DEVICES."
            )

    # Run training
    train(
        Config,
        checkpoint_dir=checkpoint_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
    )
