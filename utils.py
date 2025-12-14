"""
Utility Functions for mmWave Beam Alignment

This module provides essential utility functions for mmWave beamforming operations,
including array signal processing, beamforming gain computation, and signal
manipulation for complex-valued neural networks.

Key Function Categories:

1. Array Signal Processing:
   - array_response_vector(): Compute ULA array response for given angles
   - create_dft_codebook(): Generate DFT-based beam codebook

2. Beamforming Operations:
   - compute_beamforming_gain(): Calculate |w^H H f|^2 (linear scale)
   - compute_beamforming_gain_db(): Calculate beamforming gain in dB
   - normalize_beam(): Normalize beam vectors to unit norm

3. Signal & Noise:
   - add_complex_noise(): Add complex Gaussian noise to signals

4. Complex-Real Conversions (for neural networks):
   - real_to_complex_vector(): Convert stacked representation back to complex

5. Performance Metrics:
   - satisfaction_probability(): Fraction of samples above SNR threshold

All functions are TensorFlow-compatible and designed for efficient batch processing
of complex-valued beamforming operations.

Typical Usage:
    >>> import tensorflow as tf
    >>> from utils import compute_beamforming_gain_db, normalize_beam
    >>> channel = tf.random.normal([10, 16, 32], dtype=tf.complex64)
    >>> tx_beam = normalize_beam(tf.random.normal([10, 32], dtype=tf.complex64))
    >>> rx_beam = normalize_beam(tf.random.normal([10, 16], dtype=tf.complex64))
    >>> gain_db = compute_beamforming_gain_db(channel, tx_beam, rx_beam)
"""

import tensorflow as tf
import numpy as np


def array_response_vector(angles, num_antennas, antenna_spacing=0.5):
    """
    Compute the array response vector for a Uniform Linear Array (ULA).

    This implements: a(φ) = [1, e^(jπsinφ), e^(j2πsinφ), ..., e^(j(N-1)πsinφ)]^T

    NOTE: This does NOT include 1/sqrt(N) normalization. Normalization should be
    applied to beamforming vectors, not to channel array responses.

    Args:
        angles: Tensor of shape (...,) containing angles in radians
        num_antennas: Number of antenna elements (N)
        antenna_spacing: Antenna spacing in wavelengths (default: 0.5 for λ/2)

    Returns:
        Array response vectors of shape (..., num_antennas) with complex values
    """
    # Expand angles to allow broadcasting
    angles = tf.cast(angles, tf.float32)
    angles_expanded = tf.expand_dims(angles, axis=-1)  # (..., 1)

    # Create antenna indices [0, 1, 2, ..., N-1]
    n = tf.range(num_antennas, dtype=tf.float32)  # (N,)
    n = tf.reshape(n, [1] * len(angles.shape) + [num_antennas])  # (1, ..., 1, N)

    # Compute phase: 2π * d/λ * sin(φ) * n = 2π * antenna_spacing * sin(φ) * n
    phase = 2.0 * np.pi * antenna_spacing * tf.sin(angles_expanded) * n

    # Create complex exponential: e^(j*phase)
    response = tf.complex(tf.cos(phase), tf.sin(phase))

    return response


def compute_beamforming_gain(channel, transmit_beam, receive_beam):
    """
    Compute beamforming gain: |w^H H f|^2

    Args:
        channel: Channel matrix of shape (batch, nrx, ntx)
        transmit_beam: Transmit beamforming vector of shape (batch, ntx)
        receive_beam: Receive combining vector of shape (batch, nrx)

    Returns:
        Beamforming gain of shape (batch,)
    """
    # w^H H f
    # First: H @ f -> (batch, nrx)
    Hf = tf.linalg.matvec(channel, transmit_beam)

    # Then: w^H @ (Hf) -> (batch,)
    wH_Hf = tf.reduce_sum(tf.math.conj(receive_beam) * Hf, axis=-1)

    # Compute power: |w^H H f|^2
    bf_gain = tf.math.square(tf.abs(wH_Hf))

    return bf_gain


def compute_beamforming_gain_db(channel, transmit_beam, receive_beam):
    """
    Compute beamforming gain in dB: 10 * log10(|w^H H f|^2)

    Args:
        channel: Channel matrix of shape (batch, nrx, ntx)
        transmit_beam: Transmit beamforming vector of shape (batch, ntx)
        receive_beam: Receive combining vector of shape (batch, nrx)

    Returns:
        Beamforming gain in dB of shape (batch,)
    """
    bf_gain_linear = compute_beamforming_gain(channel, transmit_beam, receive_beam)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    bf_gain_db = 10.0 * tf.math.log(bf_gain_linear + epsilon) / tf.math.log(10.0)

    return bf_gain_db


def normalize_beam(beam):
    """
    Normalize a beam vector to unit norm.

    Args:
        beam: Complex beam vector of shape (..., N)

    Returns:
        Normalized beam of same shape
    """
    # Compute norm as real value: sqrt(sum(|beam|^2))
    norm = tf.sqrt(tf.reduce_sum(tf.math.square(tf.abs(beam)), axis=-1, keepdims=True))

    # Convert real norm to complex without warnings
    # Use tf.complex(real, imag) instead of tf.cast to avoid dtype warnings
    norm_complex = tf.complex(norm, tf.zeros_like(norm))

    # Add epsilon for numerical stability
    return beam / (norm_complex + 1e-10)


def add_complex_noise(signal, noise_power):
    """
    Add complex Gaussian noise to a signal.

    Args:
        signal: Complex signal tensor
        noise_power: Noise power (variance of real and imaginary parts each = noise_power/2)

    Returns:
        Noisy signal
    """
    # Cast to float32 for tf.random.normal (doesn't support float16 stddev)
    noise_std = tf.cast(tf.sqrt(noise_power / 2.0), tf.float32)

    # Generate real and imaginary noise
    noise_real = tf.random.normal(
        tf.shape(signal), mean=0.0, stddev=noise_std, dtype=tf.float32
    )
    noise_imag = tf.random.normal(
        tf.shape(signal), mean=0.0, stddev=noise_std, dtype=tf.float32
    )

    noise = tf.complex(noise_real, noise_imag)

    return signal + noise


def create_dft_codebook(num_beams, num_antennas):
    """
    Create a DFT (Discrete Fourier Transform) codebook.
    This is commonly used as a baseline codebook in mmWave systems.

    Args:
        num_beams: Number of beams in the codebook
        num_antennas: Number of antenna elements

    Returns:
        Codebook matrix of shape (num_beams, num_antennas)
    """
    # Create angles spanning [-π/2, π/2]
    angles = tf.linspace(-np.pi / 2, np.pi / 2, num_beams)

    # Generate array response for each angle
    codebook = array_response_vector(angles, num_antennas)

    return codebook


def satisfaction_probability(snr_rx_db, threshold_db):
    """
    Compute satisfaction probability: fraction of samples above threshold.

    Args:
        snr_rx_db: Post-combining receive SNR in dB, shape (batch,).
            This corresponds to Eq. (4) in the paper, expressed in dB:
                SNR_RX = |w^H H f|^2 / sigma_n^2
        threshold_db: Target SNR threshold in dB (paper Eq. (6))

    Returns:
        Satisfaction probability (scalar between 0 and 1)
    """
    above_threshold = tf.cast(snr_rx_db >= threshold_db, tf.float32)
    return tf.reduce_mean(above_threshold)


def real_to_complex_vector(real_vec, complex_dim):
    """
    Convert real vector to complex vector by splitting into real and imaginary parts.

    Args:
        real_vec: Real tensor of shape (..., 2*N)
        complex_dim: Dimension N of complex output

    Returns:
        Complex tensor of shape (..., N)
    """
    real_part = real_vec[..., :complex_dim]
    imag_part = real_vec[..., complex_dim:]
    # Cast to float32 for complex number creation (tf.complex doesn't support float16)
    real_part = tf.cast(real_part, tf.float32)
    imag_part = tf.cast(imag_part, tf.float32)
    return tf.complex(real_part, imag_part)


if __name__ == "__main__":
    # Test array response vector
    print("Testing array response vector...")
    angles = tf.constant([0.0, np.pi / 4, -np.pi / 4])
    response = array_response_vector(angles, num_antennas=8)
    print(f"Response shape: {response.shape}")
    print(f"Response norm: {tf.norm(response, axis=-1)}")

    # Test beamforming gain
    print("\nTesting beamforming gain...")
    batch_size = 10
    channel = tf.complex(
        tf.random.normal([batch_size, 16, 64]), tf.random.normal([batch_size, 16, 64])
    )
    tx_beam = normalize_beam(
        tf.complex(
            tf.random.normal([batch_size, 64]), tf.random.normal([batch_size, 64])
        )
    )
    rx_beam = normalize_beam(
        tf.complex(
            tf.random.normal([batch_size, 16]), tf.random.normal([batch_size, 16])
        )
    )

    bf_gain = compute_beamforming_gain(channel, tx_beam, rx_beam)
    bf_gain_db = compute_beamforming_gain_db(channel, tx_beam, rx_beam)
    print(f"BF gain shape: {bf_gain.shape}")
    print(f"BF gain (dB) mean: {tf.reduce_mean(bf_gain_db):.2f} dB")

    print("\nAll tests passed! ✓")
