import tensorflow as tf

from metrics import BeamAlignmentMetrics
from models.beam_alignment import BeamAlignmentModel
from checkpoint_utils import check_checkpoint_compatibility


# ==================== Model Loading and Evaluation ====================

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


def evaluate_at_snr_fixed_channels(model, channels, snr_db, batch_size, target_snr_db, *, start_idx=None):
    """
    Evaluate a model at a specific SNR using a fixed set of channel realizations.

    This is useful for Fig. 4-style plots where we want the SNR dependence to come
    primarily from measurement noise (and the model's decisions), not from drawing
    new channels for each SNR point.

    Args:
        model: BeamAlignmentModel instance
        channels: Fixed channel matrices (num_samples, nrx, ntx)
        snr_db: Per-antenna SNR in dB
        batch_size: Batch size
        target_snr_db: Target SNR threshold in dB (paper Eq. 6)
        start_idx: Optional fixed BS sweep start indices (num_samples,)

    Returns:
        Dictionary with metrics
    """
    metrics = BeamAlignmentMetrics(target_snr_db=target_snr_db)

    num_samples = int(channels.shape[0])
    num_batches = max(1, (num_samples + batch_size - 1) // batch_size)

    snr_linear = 10.0 ** (tf.cast(snr_db, tf.float32) / 10.0)
    noise_power = 1.0 / snr_linear  # paper per-antenna SNR: sigma_n^2 = 1/SNR_ANT

    for b in range(num_batches):
        start = b * batch_size
        end = min(num_samples, (b + 1) * batch_size)
        ch_b = channels[start:end]
        if start_idx is None:
            start_b = None
        else:
            start_b = start_idx[start:end]

        results = model.execute_beam_alignment(
            ch_b,
            noise_power,
            training=False,
            start_idx=start_b,
        )
        ch_final = results.get("channels_final", ch_b)
        metrics.update(
            ch_final,
            results["final_tx_beams"],
            results["final_rx_beams"],
            noise_power=noise_power,
        )

    return metrics.result()


def evaluate_at_snr_fixed_channels_with_ablation(
    model,
    channels,
    snr_db,
    batch_size,
    target_snr_db,
    *,
    start_idx=None,
    measurement_ablation="none",
):
    """
    Same as evaluate_at_snr_fixed_channels(), but applies a measurement ablation
    inside the sensing loop (see BeamAlignmentModel.execute_beam_alignment()).
    """
    metrics = BeamAlignmentMetrics(target_snr_db=target_snr_db)

    num_samples = int(channels.shape[0])
    num_batches = max(1, (num_samples + batch_size - 1) // batch_size)

    snr_linear = 10.0 ** (tf.cast(snr_db, tf.float32) / 10.0)
    noise_power = 1.0 / snr_linear

    for b in range(num_batches):
        start = b * batch_size
        end = min(num_samples, (b + 1) * batch_size)
        ch_b = channels[start:end]
        if start_idx is None:
            start_b = None
        else:
            start_b = start_idx[start:end]
        results = model.execute_beam_alignment(
            ch_b,
            noise_power,
            training=False,
            start_idx=start_b,
            measurement_ablation=measurement_ablation,
        )
        ch_final = results.get("channels_final", ch_b)
        metrics.update(
            ch_final,
            results["final_tx_beams"],
            results["final_rx_beams"],
            noise_power=noise_power,
        )

    return metrics.result()
