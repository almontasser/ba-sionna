"""
Graph-compiled training and validation steps.
"""

from __future__ import annotations

import tensorflow as tf

from config import Config
from metrics import compute_loss
from utils import compute_beamforming_gain_db


def sample_snr(config):
    """
    Sample a random SNR value for domain randomization.
    """
    if config.SNR_TRAIN_RANDOMIZE:
        return tf.random.uniform(
            [],
            minval=config.SNR_TRAIN_RANGE[0],
            maxval=config.SNR_TRAIN_RANGE[1],
            dtype=tf.float32,
        )
    return tf.constant(config.SNR_TRAIN, dtype=tf.float32)


@tf.function(reduce_retracing=True)
def train_step(model, optimizer, batch_size, snr_db, channels=None):
    """
    Execute one training step with domain randomization.
    """

    snr_db = tf.cast(snr_db, tf.float32)

    def _sanitize_grad(g):
        if g is None:
            return None
        if isinstance(g, tf.IndexedSlices):
            finite = tf.math.is_finite(g.values)
            values = tf.where(finite, g.values, tf.zeros_like(g.values))
            return tf.IndexedSlices(values, g.indices, g.dense_shape)
        finite = tf.math.is_finite(g)
        return tf.where(finite, g, tf.zeros_like(g))

    with tf.GradientTape() as tape:
        if channels is None:
            results = model(batch_size=batch_size, snr_db=snr_db, training=True)
            channels_final = results["channels"]
        else:
            snr_linear = tf.pow(tf.constant(10.0, tf.float32), snr_db / 10.0)
            noise_power = 1.0 / snr_linear
            results = model.execute_beam_alignment(channels, noise_power, training=True)
            channels_final = results["channels_final"]

        loss = compute_loss(
            results["beamforming_gain"],
            channels_final,
            loss_type=getattr(Config, "LOSS_TYPE", "paper"),
        )

    bf_gain_db = tf.reduce_mean(
        compute_beamforming_gain_db(
            channels_final,
            results["final_tx_beams"],
            results["final_rx_beams"],
        )
    )

    loss_finite = tf.math.is_finite(loss)
    bf_gain_db = tf.where(tf.math.is_finite(bf_gain_db), bf_gain_db, tf.zeros_like(bf_gain_db))

    def _apply_update():
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients = [_sanitize_grad(g) for g in gradients]
        gradient_norm = tf.linalg.global_norm(gradients)
        gradient_norm = tf.where(
            tf.math.is_finite(gradient_norm), gradient_norm, tf.zeros_like(gradient_norm)
        )
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, bf_gain_db, gradient_norm

    def _skip_update():
        tf.print("WARNING: non-finite loss; skipping optimizer step")
        return tf.zeros_like(loss), bf_gain_db, tf.zeros([], dtype=tf.float32)

    return tf.cond(loss_finite, _apply_update, _skip_update)


@tf.function(reduce_retracing=True)
def validate_step(model, batch_size, snr_db, channels=None):
    """Single validation step (graph-compiled for speed)."""
    snr_db = tf.cast(snr_db, tf.float32)
    if channels is None:
        results = model(batch_size=batch_size, snr_db=snr_db, training=False)
        channels_final = results["channels"]
        noise_power = results.get("noise_power", None)
    else:
        snr_linear = tf.pow(tf.constant(10.0, tf.float32), snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        results = model.execute_beam_alignment(channels, noise_power, training=False)
        channels_final = results["channels_final"]

    loss = compute_loss(
        results["beamforming_gain"],
        channels_final,
        loss_type=getattr(Config, "LOSS_TYPE", "paper"),
    )

    bf_gain_db = compute_beamforming_gain_db(
        channels_final,
        results["final_tx_beams"],
        results["final_rx_beams"],
    )

    return loss, bf_gain_db, noise_power


def validate(model, num_val_batches, batch_size, snr_db, target_snr_db):
    """
    Validate the model.
    """
    num_val_batches = max(2, min(num_val_batches, 3))
    val_batch_size = batch_size * 2

    total_loss = 0.0
    all_bf_gains_db = []
    noise_power = None

    for _ in range(num_val_batches):
        channels = None
        if getattr(Config, "TRAIN_CHANNELS_OUTSIDE_GRAPH", False):
            channels, _, _ = model.generate_channels(val_batch_size)
        loss, bf_gain_db, noise_power = validate_step(
            model, val_batch_size, snr_db, channels=channels
        )
        total_loss += float(loss.numpy())
        all_bf_gains_db.append(bf_gain_db)

    all_bf_gains_db = tf.concat(all_bf_gains_db, axis=0)

    mean_bf_gain = tf.reduce_mean(all_bf_gains_db)
    std_bf_gain = tf.math.reduce_std(all_bf_gains_db)

    # Satisfaction probability uses post-combining SNR_RX(dB) = gain_dB - 10log10(noise_power)
    if noise_power is None:
        # Fallback: interpret snr_db as per-antenna SNR, per paper Eq. (4)
        snr_linear = 10.0 ** (tf.cast(snr_db, tf.float32) / 10.0)
        noise_power = 1.0 / snr_linear

    noise_power = tf.cast(noise_power, all_bf_gains_db.dtype)
    noise_power_db = 10.0 * tf.math.log(noise_power + 1e-20) / tf.math.log(10.0)
    snr_rx_db = all_bf_gains_db - tf.cast(noise_power_db, all_bf_gains_db.dtype)
    satisfaction_prob = tf.reduce_mean(tf.cast(snr_rx_db >= target_snr_db, tf.float32))

    return {
        "val_loss": total_loss / num_val_batches,
        "mean_bf_gain_db": float(mean_bf_gain.numpy()),
        "std_bf_gain_db": float(std_bf_gain.numpy()),
        "satisfaction_prob": float(satisfaction_prob.numpy()),
    }
