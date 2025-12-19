import os
import re
import tensorflow as tf

from metrics import BeamAlignmentMetrics
from models.beam_alignment import BeamAlignmentModel
from checkpoint_utils import check_checkpoint_compatibility


# ==================== Model Loading and Evaluation ====================

def _resolve_checkpoint_dir(checkpoint_dir: str) -> str:
    if os.path.isdir(checkpoint_dir):
        return checkpoint_dir
    parent = os.path.dirname(checkpoint_dir) or "."
    prefix = os.path.basename(checkpoint_dir)
    if not os.path.isdir(parent):
        return checkpoint_dir
    pattern = re.compile(rf"^{re.escape(prefix)}")
    matches = []
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if os.path.isdir(path) and pattern.match(name):
            matches.append(path)
    if not matches:
        return checkpoint_dir
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = matches[0]
    if chosen != checkpoint_dir:
        print(f"Checkpoint dir '{checkpoint_dir}' not found; using '{chosen}'")
    return chosen


def load_c3_model(
    config,
    checkpoint_dir,
    *,
    num_sensing_steps=None,
    scenarios=None,
):
    """
    Load a C3-only BeamAlignmentModel from a training checkpoint.

    Notes:
      - This repo saves **model weights** (not optimizer state) for evaluation.
      - `scenarios` can be overridden at evaluation time to compare channel variants.
    """
    if num_sensing_steps is None:
        num_sensing_steps = config.T
    if scenarios is None:
        scenarios = config.SCENARIOS

    model = BeamAlignmentModel(
        num_tx_antennas=config.NTX,
        num_rx_antennas=config.NRX,
        codebook_size=config.NCB,
        num_sensing_steps=int(num_sensing_steps),
        rnn_hidden_size=config.RNN_HIDDEN_SIZE,
        rnn_type=config.RNN_TYPE,
        num_feedback=config.NUM_FEEDBACK,
        start_beam_index=getattr(config, "START_BEAM_INDEX", 0),
        random_start=getattr(config, "RANDOM_START", True),
        carrier_frequency=config.CARRIER_FREQUENCY,
        scenarios=scenarios,
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

    # Build model variables without generating TR 38.901 channels.
    model.build(None)

    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)

    # Restore model weights only (training checkpoints may also include optimizer).
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        ckpt_path = ckpt_manager.latest_checkpoint
        compat = check_checkpoint_compatibility(
            model,
            ckpt_path,
            require_all_trainable=True,
            require_ue_rnn_kernel=True,
        )
        if not compat.ok:
            raise ValueError(
                "Checkpoint is incompatible with the current model definition.\n"
                f"Checkpoint: {ckpt_path}\n"
                "Details:\n"
                f"{compat.summary()}\n"
                "Tip: retrain with the current config (or point to a matching checkpoint_dir)."
            )
        checkpoint.restore(ckpt_path).expect_partial()
    else:
        raise FileNotFoundError(
            f"No checkpoint found in '{checkpoint_dir}'. "
            f"Train first (e.g. python train.py -T {num_sensing_steps})."
        )

    return model


def evaluate_at_snr(model, snr_db, num_samples, batch_size, target_snr_db):
    """
    Evaluate a model at specific SNR.

    Args:
        model: BeamAlignmentModel instance
        snr_db: SNR per antenna in dB
        num_samples: Number of samples to evaluate
        batch_size: Batch size
        target_snr_db: Target SNR for satisfaction probability

    Returns:
        Dictionary with metrics
    """
    metrics = BeamAlignmentMetrics(target_snr_db=target_snr_db)
    num_batches = max(1, num_samples // batch_size)

    for _ in range(num_batches):
        results = model(batch_size=batch_size, snr_db=snr_db, training=False)
        metrics.update(
            results["channels"],
            results["final_tx_beams"],
            results["final_rx_beams"],
            noise_power=results.get("noise_power", None),
        )

    return metrics.result()


def evaluate_at_snr_with_ablation(
    model,
    snr_db,
    num_samples,
    batch_size,
    target_snr_db,
    *,
    measurement_ablation="none",
):
    """
    Evaluate a model at a specific SNR with a measurement ablation, generating
    fresh channels each batch (no caching).
    """
    metrics = BeamAlignmentMetrics(target_snr_db=target_snr_db)
    num_batches = max(1, num_samples // batch_size)

    snr_linear = 10.0 ** (tf.cast(snr_db, tf.float32) / 10.0)
    noise_power = 1.0 / snr_linear

    for _ in range(num_batches):
        channels, _, _ = model.generate_channels(batch_size)
        results = model.execute_beam_alignment(
            channels,
            noise_power,
            training=False,
            measurement_ablation=measurement_ablation,
            snr_db=snr_db,
        )
        ch_final = results.get("channels_final", channels)
        metrics.update(
            ch_final,
            results["final_tx_beams"],
            results["final_rx_beams"],
            noise_power=noise_power,
        )

    return metrics.result()
