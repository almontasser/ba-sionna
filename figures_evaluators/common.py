import tensorflow as tf

from metrics import BeamAlignmentMetrics
from models.beam_alignment import BeamAlignmentModel


# ==================== Model Loading and Evaluation ====================

def load_c3_model(
    config,
    checkpoint_dir,
    *,
    num_sensing_steps=None,
    cdl_models=None,
):
    """
    Load a C3-only BeamAlignmentModel from a training checkpoint.

    Notes:
      - Training checkpoints store both model and optimizer.
      - We initialize optimizer variables once before restore to avoid missing-slot issues.
      - `cdl_models` can be overridden at evaluation time to compare channel variants.
    """
    if num_sensing_steps is None:
        num_sensing_steps = config.T
    if cdl_models is None:
        cdl_models = config.CDL_MODELS

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
        cdl_models=cdl_models,
        delay_spread_range=config.DELAY_SPREAD_RANGE,
        ue_speed_range=config.UE_SPEED_RANGE,
    )

    # Build model variables
    _ = model(batch_size=2, snr_db=config.SNR_TRAIN, training=False)

    # Restore model weights only (training checkpoints may also include optimizer).
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
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
            results["channels"], results["final_tx_beams"], results["final_rx_beams"]
        )

    return metrics.result()
