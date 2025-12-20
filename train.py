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


def _parse_curriculum_epochs(spec: str) -> list[int]:
    epochs: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = int(part)
        except ValueError as e:
            raise ValueError(
                f"Invalid --curriculum_epochs item '{part}'. Expected integers, e.g. '3,3,94'."
            ) from e
        if val <= 0:
            raise ValueError(f"Curriculum epoch counts must be positive, got {val}.")
        epochs.append(val)
    if not epochs:
        raise ValueError("--curriculum_epochs must contain at least one positive integer.")
    return epochs


def _parse_curriculum_weights(spec: str) -> list[str]:
    items: list[str] = []
    for part in str(spec).split(";"):
        part = part.strip()
        if not part:
            continue
        items.append(part)
    if not items:
        raise ValueError(
            "--curriculum_weights must contain at least one weight spec, e.g. "
            '"UMi=1,UMa=0,RMa=0;UMi=0.7,UMa=0.2,RMa=0.1;UMi=0.333,UMa=0.333,RMa=0.333".'
        )
    return items


def _parse_scenario_curriculum(
    scenarios: list[str],
    *,
    curriculum_epochs: str,
    curriculum_weights: str,
    total_epochs: int,
) -> list[tuple[int, dict[str, float]]]:
    """
    Parse a multi-phase scenario-weight curriculum.

    Returns a list of (num_epochs_in_phase, normalized_weights_dict).
    """
    epochs = _parse_curriculum_epochs(curriculum_epochs)
    weights_specs = _parse_curriculum_weights(curriculum_weights)
    if len(weights_specs) != len(epochs):
        raise ValueError(
            f"--curriculum_epochs has {len(epochs)} phase(s) but --curriculum_weights has {len(weights_specs)}; "
            "they must match."
        )

    total = int(sum(epochs))
    if total > int(total_epochs):
        raise ValueError(
            f"Curriculum epochs sum to {total}, which exceeds --epochs={int(total_epochs)}."
        )
    if total < int(total_epochs):
        # Extend the last phase to cover the remaining epochs.
        epochs[-1] += int(total_epochs) - total

    phases: list[tuple[int, dict[str, float]]] = []
    for ep, spec in zip(epochs, weights_specs):
        w = _normalize_scenario_weights(
            list(scenarios),
            spec=spec,
            w_umi=None,
            w_uma=None,
            w_rma=None,
        )
        phases.append((int(ep), w))
    return phases


def _create_scenario_channel_model(
    config,
    scenario: str,
    *,
    generation_device: str | None = None,
) -> SionnaScenarioChannelModel:
    scenario = _canonical_scenario_name(scenario)
    if generation_device is None:
        generation_device = getattr(config, "CHANNEL_GENERATION_DEVICE", "auto")
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
        generation_device=str(generation_device),
    )


def _resolve_channel_device(config) -> str:
    return _resolve_device_request(
        getattr(config, "CHANNEL_GENERATION_DEVICE", "auto"),
        purpose="training channel generation",
    )


def _resolve_device_request(req: str, *, purpose: str) -> str:
    req = str(req).lower()
    if req == "cpu":
        return "/CPU:0"
    if req == "gpu":
        if tf.config.list_physical_devices("GPU"):
            return "/GPU:0"
        print(
            f"WARNING: {purpose} device 'gpu' requested but no GPU is visible; using CPU."
        )
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
    datasets, scenarios = _build_per_scenario_datasets(
        config,
        batch_size=batch_size,
        num_time_samples=num_time_samples,
        sampling_frequency=sampling_frequency,
    )
    seed = int(getattr(config, "RANDOM_SEED", 0) or 0)
    active_scenarios = [
        s for s in scenarios if float(scenario_weights.get(s, 0.0)) > 0.0
    ]
    mixed = _mix_scenario_datasets(
        datasets,
        scenarios,
        scenario_weights=scenario_weights,
        seed=seed,
    )
    return mixed, active_scenarios


def _build_per_scenario_datasets(
    config,
    *,
    batch_size: int,
    num_time_samples: int,
    sampling_frequency: float,
) -> tuple[list[tf.data.Dataset], list[str]]:
    """
    Build per-scenario infinite datasets, each yielding (channels, scenario_id) batches.
    """
    # Preserve the user's scenario order.
    scenarios = [_canonical_scenario_name(s) for s in getattr(config, "SCENARIOS", ["UMi", "UMa", "RMa"])]
    scenarios = list(scenarios)
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
        ds = ds.prefetch(1)
        datasets.append(ds)

    return datasets, scenarios


def _mix_scenario_datasets(
    datasets: list[tf.data.Dataset],
    scenarios: list[str],
    *,
    scenario_weights: dict[str, float],
    seed: int,
) -> tf.data.Dataset:
    ds_list: list[tf.data.Dataset] = []
    weights: list[float] = []
    for ds, scenario in zip(datasets, scenarios):
        w = float(scenario_weights.get(scenario, 0.0))
        if w <= 0.0:
            continue
        ds_list.append(ds)
        weights.append(w)
    if not ds_list:
        raise ValueError("Scenario weights must have at least one value > 0.")
    total = float(sum(weights))
    weights = [w / total for w in weights]
    return tf.data.Dataset.sample_from_datasets(ds_list, weights=weights, seed=int(seed))


def _run_lr_range_test(
    model,
    config,
    *,
    summary_writer,
    scenario_weights: dict[str, float],
    lr_min: float,
    lr_max: float,
    lr_steps: int,
    stop_factor: float,
    seed: int,
) -> None:
    """
    Run a short LR range test (exponential LR increase) and log loss vs LR.
    """
    lr_min = float(lr_min)
    lr_max = float(lr_max)
    lr_steps = int(lr_steps)
    stop_factor = float(stop_factor)

    if lr_min <= 0.0 or lr_max <= 0.0:
        raise ValueError("--lr_range_min_lr/--lr_range_max_lr must be > 0.")
    if lr_max <= lr_min:
        raise ValueError("--lr_range_max_lr must be greater than --lr_range_min_lr.")
    if lr_steps < 10:
        raise ValueError("--lr_range_steps must be >= 10.")

    print("\n" + "=" * 80)
    print("LEARNING-RATE RANGE TEST")
    print("=" * 80)
    print(f"LR range: {lr_min:g} -> {lr_max:g} over {lr_steps} steps")
    print(f"Early stop factor: {stop_factor:g}x")

    # Build per-scenario dataset (one scenario per batch) if needed.
    num_time_samples = 1
    sampling_frequency = 1.0
    if getattr(config, "MOBILITY_ENABLE", False):
        nts = getattr(config, "MOBILITY_NUM_TIME_SAMPLES", None)
        num_time_samples = int(nts) if nts is not None else int(config.T + 1)
        sampling_frequency = float(getattr(config, "MOBILITY_SAMPLING_FREQUENCY_HZ", 1.0))

    use_scenario_ds = len(getattr(config, "SCENARIOS", [])) > 1
    train_scenario_iter = None
    if use_scenario_ds:
        train_scenario_ds, active = _build_per_scenario_batch_dataset(
            config,
            batch_size=int(config.BATCH_SIZE),
            num_time_samples=int(num_time_samples),
            sampling_frequency=float(sampling_frequency),
            scenario_weights=scenario_weights,
        )
        train_scenario_iter = iter(train_scenario_ds)
        print("\nLR range test scenarios (per-batch sampling):")
        for s in active:
            print(f"  {s}: {scenario_weights.get(s, 0.0):.6f}")

    # Optimizer with manually controlled LR (assign per-step).
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_min)

    # Exponential LR sweep.
    lr_ratio = lr_max / lr_min
    lr_mult = lr_ratio ** (1.0 / float(lr_steps - 1))

    beta = 0.98
    avg_loss = 0.0
    best_loss = float("inf")
    best_lr = lr_min

    pbar = tqdm(range(lr_steps), desc="LR range test")
    for step in pbar:
        lr_now = lr_min * (lr_mult ** step)
        try:
            optimizer.learning_rate.assign(lr_now)
        except Exception:
            optimizer.learning_rate = lr_now

        if use_scenario_ds:
            if train_scenario_iter is None:
                raise RuntimeError("Per-scenario LR test requested but no dataset iterator is available.")
            channels, _scenario_id = next(train_scenario_iter)
        elif getattr(config, "TRAIN_CHANNELS_OUTSIDE_GRAPH", False):
            channels, _, _ = model.generate_channels(int(config.BATCH_SIZE))
        else:
            channels = None

        snr_db = sample_snr(config)
        loss, bf_gain_db, grad_norm, update_skipped = train_step(
            model,
            optimizer,
            config.BATCH_SIZE,
            snr_db,
            channels=channels,
        )

        loss_val = float(loss.numpy())
        grad_norm_val = float(grad_norm.numpy())
        bf_gain_val = float(bf_gain_db.numpy())
        skipped_val = int(update_skipped.numpy())

        if not np.isfinite(loss_val) or skipped_val:
            print(
                f"\nStopped LR range test at step {step} due to non-finite loss "
                f"(skipped={skipped_val})."
            )
            break

        avg_loss = beta * avg_loss + (1.0 - beta) * loss_val
        smooth_loss = avg_loss / (1.0 - beta ** (step + 1))

        if smooth_loss < best_loss:
            best_loss = smooth_loss
            best_lr = float(lr_now)

        if step > 10 and stop_factor > 1.0:
            # Robust early-stop criterion that works for negative losses:
            # stop when smoothed loss is worse than best by ~stop_factor*|best|.
            threshold = best_loss + abs(best_loss) * (stop_factor - 1.0)
            if smooth_loss > threshold:
                print(
                    f"\nStopped LR range test at step {step}: loss exploded "
                    f"({smooth_loss:.4f} > {threshold:.4f} threshold)."
                )
                break

        pbar.set_postfix(
            {
                "lr": f"{lr_now:.2e}",
                "loss": f"{loss_val:.4f}",
                "smooth": f"{smooth_loss:.4f}",
                "bf_gain": f"{bf_gain_val:.2f} dB",
            }
        )

        with summary_writer.as_default():
            tf.summary.scalar("lr_range_test/loss", loss, step=step)
            tf.summary.scalar("lr_range_test/smoothed_loss", smooth_loss, step=step)
            tf.summary.scalar(
                "lr_range_test/learning_rate",
                tf.constant(lr_now, dtype=tf.float32),
                step=step,
            )
            tf.summary.scalar("lr_range_test/bf_gain_db", bf_gain_db, step=step)
            tf.summary.scalar("lr_range_test/gradient_norm", grad_norm, step=step)
            tf.summary.scalar(
                "lr_range_test/update_skipped", tf.cast(update_skipped, tf.float32), step=step
            )

    print("\nLR range test complete.")
    print(f"Best smoothed loss: {best_loss:.4f} at lr={best_lr:.3g}")
    print(
        "Suggested initial LR: choose a value below the divergence point, "
        "often around the best-loss LR."
    )


def train(
    config,
    checkpoint_dir=None,
    log_dir=None,
    *,
    run_name=None,
    resume: bool = True,
    reset_optimizer: bool = False,
    scenario_curriculum_phases: list[tuple[int, dict[str, float]]] | None = None,
    lr_range_test: bool = False,
    lr_range_min_lr: float = 1e-5,
    lr_range_max_lr: float = 1e-2,
    lr_range_steps: int = 2000,
    lr_range_stop_factor: float = 4.0,
):
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

    if scenario_curriculum_phases is not None:
        print("\nScenario curriculum (single run):")
        for pi, (num_epochs, weights) in enumerate(scenario_curriculum_phases, start=1):
            parts = []
            for s in ["UMi", "UMa", "RMa"]:
                if s in weights:
                    parts.append(f"{s}={weights[s]:.3f}")
            print(f"  Phase {pi}: {int(num_epochs)} epochs: {', '.join(parts)}")
        with summary_writer.as_default():
            tf.summary.text(
                "run/scenario_curriculum",
                tf.constant(str(scenario_curriculum_phases)),
                step=0,
            )

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

        # Optional: LR range test (then exit).
        if lr_range_test:
            _run_lr_range_test(
                model,
                config,
                summary_writer=summary_writer,
                scenario_weights=scenario_weights,
                lr_min=lr_range_min_lr,
                lr_max=lr_range_max_lr,
                lr_steps=lr_range_steps,
                stop_factor=lr_range_stop_factor,
                seed=seed,
            )
            return

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

        # Setup checkpoint manager.
        epoch_var = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch")
        checkpoint = tf.train.Checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch_var,
            lr_scale=lr_schedule.scale,
        )
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

        def _set_save_counter_from_path(ckpt_path: str) -> None:
            """Avoid clobbering existing ckpt-N when restoring model-only."""
            base = os.path.basename(str(ckpt_path))
            if "-" not in base:
                return
            try:
                n = int(base.split("-")[-1])
            except ValueError:
                return
            try:
                checkpoint.save_counter.assign(n)
            except Exception:
                pass

        # Restore from checkpoint if available and compatible.
        start_epoch = 0
        if not bool(resume):
            if checkpoint_manager.latest_checkpoint:
                fresh_dir = f"{checkpoint_dir}_fresh_{timestamp}"
                print(
                    "Resume disabled: not restoring from existing checkpoints.\n"
                    f"Using fresh checkpoint dir: {fresh_dir}"
                )
                os.makedirs(fresh_dir, exist_ok=True)
                checkpoint_dir = fresh_dir
                checkpoint_manager = tf.train.CheckpointManager(
                    checkpoint, checkpoint_dir, max_to_keep=5
                )
            print("Starting training from scratch (resume disabled)")
        elif checkpoint_manager.latest_checkpoint:
            ckpt_path = checkpoint_manager.latest_checkpoint
            compat = check_checkpoint_compatibility(
                model,
                ckpt_path,
                require_all_trainable=True,
                require_ue_rnn_kernel=True,
            )
            if compat.ok:
                try:
                    if bool(reset_optimizer):
                        # Restore weights (and epoch if present) but keep optimizer/LR fresh.
                        tf.train.Checkpoint(model=model, epoch=epoch_var).restore(
                            ckpt_path
                        ).expect_partial()
                        start_epoch = int(epoch_var.numpy())
                        _set_save_counter_from_path(ckpt_path)
                        print(
                            f"✓ Restored model weights from {ckpt_path} "
                            f"(optimizer reset; start_epoch={start_epoch})"
                        )
                    else:
                        checkpoint.restore(ckpt_path).expect_partial()
                        start_epoch = int(epoch_var.numpy())
                        print(
                            f"✓ Restored model+optimizer from {ckpt_path} (start_epoch={start_epoch})"
                        )
                except (ValueError, tf.errors.InvalidArgumentError) as e:
                    if not bool(reset_optimizer):
                        # Common case: model variables match, but optimizer slots/schedule mismatch.
                        # Fall back to model-only restore so training can proceed.
                        print(
                            "WARNING: Full checkpoint restore failed; attempting model-only restore with optimizer reset.\n"
                            f"Restore error: {e}"
                        )
                        try:
                            tf.train.Checkpoint(model=model, epoch=epoch_var).restore(
                                ckpt_path
                            ).expect_partial()
                            start_epoch = int(epoch_var.numpy())
                            _set_save_counter_from_path(ckpt_path)
                            print(
                                f"✓ Restored model weights from {ckpt_path} "
                                f"(optimizer reset; start_epoch={start_epoch})"
                            )
                        except (ValueError, tf.errors.InvalidArgumentError) as e2:
                            print(
                                "WARNING: Model-only restore also failed.\n"
                                "Starting training from scratch.\n"
                                f"Restore error: {e2}"
                            )
                            fresh_dir = f"{checkpoint_dir}_fresh_{timestamp}"
                            print(f"Using fresh checkpoint dir: {fresh_dir}")
                            os.makedirs(fresh_dir, exist_ok=True)
                            checkpoint_dir = fresh_dir
                            checkpoint_manager = tf.train.CheckpointManager(
                                checkpoint, checkpoint_dir, max_to_keep=5
                            )
                            epoch_var.assign(0)
                            start_epoch = 0
                    else:
                        print(
                            "WARNING: Model restore failed due to incompatibility.\n"
                            "Starting training from scratch.\n"
                            f"Restore error: {e}"
                        )
                        fresh_dir = f"{checkpoint_dir}_fresh_{timestamp}"
                        print(f"Using fresh checkpoint dir: {fresh_dir}")
                        os.makedirs(fresh_dir, exist_ok=True)
                        checkpoint_dir = fresh_dir
                        checkpoint_manager = tf.train.CheckpointManager(
                            checkpoint, checkpoint_dir, max_to_keep=5
                        )
                        epoch_var.assign(0)
                        start_epoch = 0
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
                epoch_var.assign(0)
                start_epoch = 0
        else:
            print("Starting training from scratch")

        if start_epoch >= int(config.EPOCHS):
            print(
                f"Checkpoint indicates epoch={start_epoch}, which is >= requested --epochs={int(config.EPOCHS)}.\n"
                "Nothing to do."
            )
            return
        
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

        use_scenario_ds = len(getattr(config, "SCENARIOS", [])) > 1

        # Curriculum schedule: map epoch -> scenario weights.
        curriculum_bounds = None
        if scenario_curriculum_phases is not None:
            curriculum_bounds = []
            cur = 0
            for num_ep, weights in scenario_curriculum_phases:
                start = int(cur)
                end = int(cur + int(num_ep))
                curriculum_bounds.append((start, end, dict(weights)))
                cur = end

        train_scenarios = [
            _canonical_scenario_name(s) for s in getattr(config, "SCENARIOS", ["UMi", "UMa", "RMa"])
        ]

        # Per-scenario tf.data pipeline: one scenario per batch, sampled by weights.
        train_scenario_iter = None
        per_scenario_datasets = None
        per_scenario_scenarios = None
        current_curriculum_phase = None
        if use_scenario_ds:
            per_scenario_datasets, per_scenario_scenarios = _build_per_scenario_datasets(
                config,
                batch_size=int(config.BATCH_SIZE),
                num_time_samples=int(num_time_samples),
                sampling_frequency=float(sampling_frequency),
            )
            train_scenarios = list(per_scenario_scenarios)
            if curriculum_bounds is None:
                train_scenario_ds = _mix_scenario_datasets(
                    per_scenario_datasets,
                    per_scenario_scenarios,
                    scenario_weights=scenario_weights,
                    seed=seed,
                )
                train_scenario_iter = iter(train_scenario_ds)
                active = [
                    s
                    for s in per_scenario_scenarios
                    if float(scenario_weights.get(s, 0.0)) > 0.0
                ]
                print("\nTraining data: per-scenario batches via tf.data.sample_from_datasets")
                for s in active:
                    print(f"  {s}: {scenario_weights.get(s, 0.0):.6f}")
            else:
                print(
                    "\nTraining data: per-scenario batches via tf.data.sample_from_datasets "
                    "(curriculum enabled)"
                )

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

            # Update scenario weights for this epoch (curriculum), and rebuild the
            # per-scenario dataset only when switching phases.
            if (
                use_scenario_ds
                and curriculum_bounds is not None
                and per_scenario_datasets is not None
                and per_scenario_scenarios is not None
            ):
                phase_idx = None
                phase_start = 0
                phase_end = 0
                phase_weights = None
                for i, (s, e, w) in enumerate(curriculum_bounds):
                    if int(s) <= int(epoch) < int(e):
                        phase_idx = int(i)
                        phase_start = int(s)
                        phase_end = int(e)
                        phase_weights = dict(w)
                        break
                if phase_idx is None or phase_weights is None:
                    raise RuntimeError(
                        f"Scenario curriculum does not cover epoch {epoch}. "
                        "Check --curriculum_epochs vs --epochs."
                    )
                if current_curriculum_phase != phase_idx:
                    current_curriculum_phase = phase_idx
                    scenario_weights = dict(phase_weights)
                    config.SCENARIO_WEIGHTS = dict(scenario_weights)
                    train_scenario_ds = _mix_scenario_datasets(
                        per_scenario_datasets,
                        per_scenario_scenarios,
                        scenario_weights=scenario_weights,
                        seed=seed + int(phase_idx),
                    )
                    train_scenario_iter = iter(train_scenario_ds)

                    active = [
                        s
                        for s in (per_scenario_scenarios or train_scenarios)
                        if float(scenario_weights.get(s, 0.0)) > 0.0
                    ]
                    print(
                        f"\nScenario curriculum phase {phase_idx + 1}: "
                        f"epochs {phase_start + 1}-{phase_end}"
                    )
                    for s in active:
                        print(f"  {s}: {scenario_weights.get(s, 0.0):.6f}")
                    with summary_writer.as_default():
                        tf.summary.scalar(
                            "train/scenario_curriculum_phase",
                            float(phase_idx + 1),
                            step=global_step,
                        )
                        for s in ["UMi", "UMa", "RMa"]:
                            if s in scenario_weights:
                                tf.summary.scalar(
                                    f"train/scenario_weight_{s}",
                                    scenario_weights[s],
                                    step=global_step,
                                )

            try:
                lr_now = float(_current_learning_rate(optimizer).numpy())
                iter_now = int(optimizer.iterations.numpy())
                print(f"LR @ step {iter_now}: {lr_now:.6g}")
            except Exception:
                pass
            
            # Training
            epoch_loss = 0.0
            epoch_bf_gain = 0.0

            if use_scenario_ds:
                if train_scenario_iter is None:
                    raise RuntimeError(
                        "Per-scenario training requested but no dataset iterator is available."
                    )
                # One scenario per batch, chosen by the configured weights (online channel generation).
                pbar = tqdm(range(steps_per_epoch), desc="Training")
                for step in pbar:
                    channels, _scenario_id = next(train_scenario_iter)
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
            
            # Validation (fresh channels each epoch; no fixed caching).
            print("Validating...")
            val_metrics = validate(
                model,
                val_batches,
                config.BATCH_SIZE,
                config.SNR_TRAIN,
                config.SNR_TARGET,
            )

            print(f"Validation - Loss: {val_metrics['val_loss']:.4f}")
            print(
                f"             BF Gain: {val_metrics['mean_bf_gain_db']:.2f} ± {val_metrics['std_bf_gain_db']:.2f} dB"
            )
            print(f"             Satisfaction Prob: {val_metrics['satisfaction_prob']:.3f}")

            # Log to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar("val/loss", val_metrics["val_loss"], step=global_step)
                tf.summary.scalar("val/bf_gain_db", val_metrics["mean_bf_gain_db"], step=global_step)
                tf.summary.scalar(
                    "val/satisfaction_prob", val_metrics["satisfaction_prob"], step=global_step
                )
            
            # Update epoch counter in the checkpoint (store "next epoch to run").
            epoch_var.assign(int(epoch) + 1)

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
        '--lr_range_test',
        action='store_true',
        help='Run a short LR range test (exponential LR sweep) and exit.',
    )
    parser.add_argument(
        '--lr_range_min_lr',
        type=float,
        default=1e-5,
        help='LR range test: minimum LR (default: 1e-5)',
    )
    parser.add_argument(
        '--lr_range_max_lr',
        type=float,
        default=1e-2,
        help='LR range test: maximum LR (default: 1e-2)',
    )
    parser.add_argument(
        '--lr_range_steps',
        type=int,
        default=2000,
        help='LR range test: number of steps (default: 2000)',
    )
    parser.add_argument(
        '--lr_range_stop_factor',
        type=float,
        default=4.0,
        help='LR range test: early-stop factor on smoothed loss (default: 4.0)',
    )
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
    parser.add_argument(
        '--resume',
        type=int,
        default=1,
        choices=[0, 1],
        help='If 1, resume from latest checkpoint in --checkpoint_dir when available (default: 1).',
    )
    parser.add_argument(
        '--reset_optimizer',
        type=int,
        default=0,
        choices=[0, 1],
        help='If 1, restore model weights but reset optimizer/LR state when resuming (default: 0).',
    )
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
    parser.add_argument(
        '--scenario_curriculum',
        action='store_true',
        help='Enable an in-run scenario-weight curriculum (weights change across epochs).',
    )
    parser.add_argument(
        '--curriculum_epochs',
        type=str,
        default=None,
        help='Comma-separated phase lengths in epochs, e.g. "3,3,94". Requires --scenario_curriculum.',
    )
    parser.add_argument(
        '--curriculum_weights',
        type=str,
        default=None,
        help=(
            'Semicolon-separated weight specs per phase, e.g. '
            '"UMi=1,UMa=0,RMa=0;UMi=0.7,UMa=0.2,RMa=0.1;UMi=0.333,UMa=0.333,RMa=0.333". '
            "Requires --scenario_curriculum."
        ),
    )
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
    if args.seed is not None:
        Config.RANDOM_SEED = int(args.seed)

    # Scenario weights (batch-level sampling) or in-run curriculum.
    scenario_curriculum_phases = None
    if args.scenario_curriculum:
        if args.scenario_weights is not None or any(
            x is not None for x in (args.w_umi, args.w_uma, args.w_rma)
        ):
            raise ValueError(
                "Use either --scenario_curriculum or --scenario_weights/--w_umi/--w_uma/--w_rma, not both."
            )
        if args.curriculum_epochs is None or args.curriculum_weights is None:
            raise ValueError(
                "--scenario_curriculum requires both --curriculum_epochs and --curriculum_weights."
            )
        scenario_curriculum_phases = _parse_scenario_curriculum(
            list(getattr(Config, "SCENARIOS", ["UMi", "UMa", "RMa"])),
            curriculum_epochs=str(args.curriculum_epochs),
            curriculum_weights=str(args.curriculum_weights),
            total_epochs=int(getattr(Config, "EPOCHS", 0)),
        )
        # Seed the initial weights for logging/config printing.
        Config.SCENARIO_WEIGHTS = dict(scenario_curriculum_phases[0][1])
    else:
        if args.curriculum_epochs is not None or args.curriculum_weights is not None:
            print(
                "WARNING: --curriculum_epochs/--curriculum_weights provided without --scenario_curriculum; ignoring."
            )
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
        resume=bool(int(args.resume)),
        reset_optimizer=bool(int(args.reset_optimizer)),
        scenario_curriculum_phases=scenario_curriculum_phases,
        lr_range_test=bool(args.lr_range_test),
        lr_range_min_lr=float(args.lr_range_min_lr),
        lr_range_max_lr=float(args.lr_range_max_lr),
        lr_range_steps=int(args.lr_range_steps),
        lr_range_stop_factor=float(args.lr_range_stop_factor),
    )
