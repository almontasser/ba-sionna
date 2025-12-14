"""
Narrowband mapping helpers for frequency-selective TR 38.901 channels.

The paper's sensing model uses a narrowband matrix H in:
  y_t = w_t^H H f_t + w_t^H n_t

However, TR 38.901 system-level channels are generated as a CIR (paths+delays),
so we explicitly map:
  (h_cir, tau) -> CFR at selected frequency offsets -> narrowband H
"""

from __future__ import annotations

import tensorflow as tf


def frequency_offsets_hz(
    narrowband_method: str,
    fft_size: int,
    subcarrier_spacing_hz: float,
    narrowband_subcarrier: int | None,
) -> tf.Tensor:
    """Return subcarrier frequency offsets relative to DC (Hz)."""
    narrowband_method = str(narrowband_method)

    if narrowband_method == "center":
        return tf.zeros([1], tf.float32)  # DC only

    if narrowband_method == "subcarrier":
        k = int(fft_size) // 2 if narrowband_subcarrier is None else int(narrowband_subcarrier)
        f0 = (float(k) - (float(fft_size) / 2.0)) * float(subcarrier_spacing_hz)
        return tf.constant([f0], tf.float32)

    if narrowband_method == "mean_cfr":
        k = tf.range(int(fft_size), dtype=tf.float32)
        return (k - (float(fft_size) / 2.0)) * float(subcarrier_spacing_hz)  # (K,)

    raise ValueError(
        f"Unknown narrowband_method '{narrowband_method}'. "
        "Use 'center', 'subcarrier', or 'mean_cfr'."
    )


def cir_to_cfr(h_cir: tf.Tensor, tau_s: tf.Tensor, f_offsets_hz: tf.Tensor) -> tf.Tensor:
    """
    Convert CIR to CFR at specified frequency offsets.

    Args:
        h_cir: Complex path coefficients with either shape:
            - (B, NRX, NTX, P) for a static snapshot
            - (B, NRX, NTX, P, S) for S time samples (mobility/time variation)
        tau_s: (B, P) delays in seconds
        f_offsets_hz: (K,) frequency offsets relative to DC in Hz

    Returns:
        h_freq: CFR with either shape:
            - (B, NRX, NTX, K)
            - (B, NRX, NTX, K, S)
    """
    two_pi = tf.constant(2.0 * 3.141592653589793, tf.float32)
    phase = -two_pi * tf.expand_dims(tau_s, axis=-1) * tf.reshape(f_offsets_hz, [1, 1, -1])
    # (B, P, K)
    exp_term = tf.exp(tf.complex(tf.zeros_like(phase), phase))  # (B, P, K)

    if h_cir.shape.rank == 4:
        return tf.einsum("b i j p, b p k -> b i j k", h_cir, exp_term)
    if h_cir.shape.rank == 5:
        return tf.einsum("b i j p s, b p k -> b i j k s", h_cir, exp_term)

    raise ValueError(
        "h_cir must have rank 4 (B,NRX,NTX,P) or rank 5 (B,NRX,NTX,P,S). "
        f"Got rank={h_cir.shape.rank}."
    )


def reduce_to_narrowband(h_freq: tf.Tensor, narrowband_method: str) -> tf.Tensor:
    """
    Reduce CFR to a narrowband H.

    Input:
      - Static: (B, NRX, NTX, K)
      - Time-varying: (B, NRX, NTX, K, S)

    Output:
      - Static: (B, NRX, NTX)
      - Time-varying: (B, NRX, NTX, S)
    """
    narrowband_method = str(narrowband_method)

    if h_freq.shape.rank == 4:
        freq_axis = -1
    elif h_freq.shape.rank == 5:
        freq_axis = -2
    else:
        raise ValueError(
            "h_freq must have rank 4 (B,NRX,NTX,K) or rank 5 (B,NRX,NTX,K,S). "
            f"Got rank={h_freq.shape.rank}."
        )

    if narrowband_method == "mean_cfr":
        return tf.reduce_mean(h_freq, axis=freq_axis)
    return tf.squeeze(h_freq, axis=freq_axis)

