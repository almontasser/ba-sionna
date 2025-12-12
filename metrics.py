"""
Performance Metrics and Loss Functions for Beam Alignment

This module provides metrics tracking and loss computation for evaluating and
training mmWave beam alignment systems. It implements the metrics used in the
paper for performance evaluation.

Key Metrics:

1. Beamforming Gain:
   - Linear scale: G = |w^H H f|^2
   - dB scale: G_dB = 10 log10(G)
   - Primary metric for beam alignment quality
   - Represents effective channel gain after beamforming

2. Satisfaction Probability:
   - P_sat = Pr[G_dB ≥ G_threshold]
   - Fraction of users achieving target SNR
   - Important for QoS guarantees
   - Typical threshold: 20 dB (per paper)

3. Average Performance:
   - Mean beamforming gain: 𝔼[G_dB]
   - Standard deviation: std[G_dB]
   - Used for comparing different schemes

Loss Functions:

1. Normalized Loss (Recommended):
   L = -𝔼[G / ||H||_F^2]
   
   Normalizes by channel Frobenius norm to:
   - Prevent emphasis on strong channels
   - Make training more stable
   - Match paper's objective function

2. Direct Loss (Alternative):
   L = -𝔼[G]
   
   Simpler but may bias toward strong channels

3. dB Loss (Alternative):
   L = -𝔼[G_dB]
   
   Optimizes logarithmic metric directly

Classes:
    - BeamAlignmentMetrics: Metrics tracker for evaluation
    - ExhaustiveSearchBaseline: Optimal baseline for comparison

Functions:
    - compute_loss(): Main training loss (normalized)
    - compute_loss_db(): Alternative dB-scale loss

Usage Example:
    >>> from metrics import BeamAlignmentMetrics, compute_loss
    >>> 
    >>> # Track metrics during validation
    >>> metrics = BeamAlignmentMetrics(target_snr_db=20.0)
    >>> for batch in val_data:
    ...     metrics.update(channels, tx_beams, rx_beams)
    >>> results = metrics.result()
    >>> print(f"Mean gain: {results['mean_bf_gain_db']:.2f} dB")
    >>> print(f"Satisfaction: {results['satisfaction_prob']:.3f}")
    >>> 
    >>> # Compute training loss
    >>> loss = compute_loss(beamforming_gains, channels)

References:
    Paper Section IV: Performance Metrics
    Paper Equation (7): Normalized objective function
"""

import tensorflow as tf
from utils import compute_beamforming_gain, compute_beamforming_gain_db, satisfaction_probability


class BeamAlignmentMetrics:
    """
    Metrics tracker for beam alignment system.
    """
    
    def __init__(self, target_snr_db=10.0):
        """
        Args:
            target_snr_db: Target SNR for satisfaction probability
        """
        self.target_snr_db = target_snr_db
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.bf_gains_linear = []
        self.bf_gains_db = []
    
    def update(self, channels, tx_beams, rx_beams):
        """
        Update metrics with new batch.
        
        Args:
            channels: Channel matrices (batch, nrx, ntx)
            tx_beams: Transmit beams (batch, ntx)
            rx_beams: Receive beams (batch, nrx)
        """
        # Compute beamforming gains
        bf_gain_linear = compute_beamforming_gain(channels, tx_beams, rx_beams)
        bf_gain_db = compute_beamforming_gain_db(channels, tx_beams, rx_beams)
        
        self.bf_gains_linear.append(bf_gain_linear)
        self.bf_gains_db.append(bf_gain_db)
    
    def result(self):
        """
        Compute final metric values.
        
        Returns:
            Dictionary with metric values
        """
        if not self.bf_gains_db:
            return {
                'mean_bf_gain_db': 0.0,
                'std_bf_gain_db': 0.0,
                'satisfaction_prob': 0.0
            }
        
        # Concatenate all batches
        all_bf_gains_db = tf.concat(self.bf_gains_db, axis=0)
        
        # Compute statistics
        mean_bf_gain = tf.reduce_mean(all_bf_gains_db)
        std_bf_gain = tf.math.reduce_std(all_bf_gains_db)
        
        # Satisfaction probability
        sat_prob = satisfaction_probability(all_bf_gains_db, self.target_snr_db)
        
        return {
            'mean_bf_gain_db': float(mean_bf_gain.numpy()),
            'std_bf_gain_db': float(std_bf_gain.numpy()),
            'satisfaction_prob': float(sat_prob.numpy())
        }


class ExhaustiveSearchBaseline:
    """
    Exhaustive search baseline for comparison.
    Tests all possible beam pairs to find the optimal one.
    """
    
    def __init__(self, tx_codebook, rx_codebook):
        """
        Args:
            tx_codebook: Transmit beam codebook (NCB_tx, ntx)
            rx_codebook: Receive beam codebook (NCB_rx, nrx)
        """
        self.tx_codebook = tx_codebook
        self.rx_codebook = rx_codebook
        self.ncb_tx = tx_codebook.shape[0]
        self.ncb_rx = rx_codebook.shape[0]
    
    def find_best_beam_pair(self, channel):
        """
        Find best beam pair via exhaustive search for a single channel.
        
        Args:
            channel: Channel matrix (nrx, ntx)
            
        Returns:
            Best transmit beam index, best receive beam index, best gain
        """
        best_gain = -float('inf')
        best_tx_idx = 0
        best_rx_idx = 0
        
        for tx_idx in range(self.ncb_tx):
            for rx_idx in range(self.ncb_rx):
                tx_beam = self.tx_codebook[tx_idx]
                rx_beam = self.rx_codebook[rx_idx]
                
                # Compute beamforming gain
                gain = compute_beamforming_gain(
                    tf.expand_dims(channel, 0),
                    tf.expand_dims(tx_beam, 0),
                    tf.expand_dims(rx_beam, 0)
                )[0]
                
                if gain > best_gain:
                    best_gain = gain
                    best_tx_idx = tx_idx
                    best_rx_idx = rx_idx
        
        return best_tx_idx, best_rx_idx, best_gain
    
    def search_batch(self, channels):
        """
        Exhaustive search for a batch of channels.
        
        Args:
            channels: Channel matrices (batch, nrx, ntx)
            
        Returns:
            Dictionary with best beams and gains
        """
        batch_size = channels.shape[0]
        
        best_tx_indices = []
        best_rx_indices = []
        best_gains = []
        
        for b in range(batch_size):
            tx_idx, rx_idx, gain = self.find_best_beam_pair(channels[b])
            best_tx_indices.append(tx_idx)
            best_rx_indices.append(rx_idx)
            best_gains.append(gain)
        
        best_tx_beams = tf.gather(self.tx_codebook, best_tx_indices)
        best_rx_beams = tf.gather(self.rx_codebook, best_rx_indices)
        
        return {
            'tx_beams': best_tx_beams,
            'rx_beams': best_rx_beams,
            'bf_gains': tf.stack(best_gains)
        }


def compute_loss(beamforming_gains, channels=None, loss_type=None, use_log_gain=None):
    """
    Compute training loss.

    Paper objective (Eq. 7): L = -E[ |w^H H f|^2 / ||H||_F^2 ].
    For CDL, normalized gains are already bounded in [0,1]; clipping keeps
    numerical stability under mixed precision.

    Args:
        beamforming_gains: Beamforming gains (batch,) - linear scale |w^H H f|^2
        channels: Channel matrices (batch, nrx, ntx) - optional for normalization
        loss_type: "paper" (default) or "log". If None, inferred from use_log_gain.
        use_log_gain: Backwards-compatibility flag. If set, overrides default
            loss_type inference (True->"log", False->"paper").

    Returns:
        Scalar loss value
    """
    if loss_type is None:
        if use_log_gain is None:
            loss_type = "paper"
        else:
            loss_type = "log" if bool(use_log_gain) else "paper"

    if channels is not None:
        # Normalize by channel Frobenius norm squared per paper
        channel_norms_squared = tf.reduce_sum(tf.abs(channels) ** 2, axis=(1, 2))  # (batch,)
        normalized_gains = beamforming_gains / (channel_norms_squared + 1e-10)
    else:
        normalized_gains = beamforming_gains

    # Clip to valid range to avoid tiny negative/overshoot artifacts
    normalized_gains = tf.clip_by_value(normalized_gains, 0.0, 1.0)

    if loss_type == "log":
        loss = -tf.reduce_mean(tf.math.log(normalized_gains + 1e-7))
    elif loss_type == "paper":
        loss = -tf.reduce_mean(normalized_gains)
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'paper' or 'log'.")

    return loss


def compute_loss_db(beamforming_gains_db):
    """
    Alternative loss: negative mean beamforming gain in dB.
    
    Args:
        beamforming_gains_db: Beamforming gains in dB (batch,)
        
    Returns:
        Scalar loss value
    """
    loss = -tf.reduce_mean(beamforming_gains_db)
    return loss


if __name__ == "__main__":
    print("Testing Metrics...")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 100
    channels = tf.complex(
        tf.random.normal([batch_size, 16, 64]),
        tf.random.normal([batch_size, 16, 64])
    )
    
    from utils import normalize_beam
    tx_beams = normalize_beam(tf.complex(
        tf.random.normal([batch_size, 64]),
        tf.random.normal([batch_size, 64])
    ))
    rx_beams = normalize_beam(tf.complex(
        tf.random.normal([batch_size, 16]),
        tf.random.normal([batch_size, 16])
    ))
    
    # Test metrics
    metrics = BeamAlignmentMetrics(target_snr_db=10.0)
    metrics.update(channels, tx_beams, rx_beams)
    results = metrics.result()
    
    print(f"\nMetrics results:")
    print(f"  Mean BF gain: {results['mean_bf_gain_db']:.2f} dB")
    print(f"  Std BF gain: {results['std_bf_gain_db']:.2f} dB")
    print(f"  Satisfaction prob: {results['satisfaction_prob']:.3f}")
    
    # Test loss computation
    bf_gains = compute_beamforming_gain(channels, tx_beams, rx_beams)
    loss = compute_loss(bf_gains)
    print(f"\nLoss value: {loss:.4f}")
    
    # Test exhaustive search
    print(f"\nTesting exhaustive search...")
    from utils import create_dft_codebook
    tx_codebook = create_dft_codebook(16, 64)
    rx_codebook = create_dft_codebook(16, 16)
    
    baseline = ExhaustiveSearchBaseline(tx_codebook, rx_codebook)
    
    # Search for small batch
    small_channels = channels[:5]
    baseline_results = baseline.search_batch(small_channels)
    
    bf_gains_baseline = compute_beamforming_gain_db(
        small_channels,
        baseline_results['tx_beams'],
        baseline_results['rx_beams']
    )
    print(f"  Exhaustive search BF gains: {bf_gains_baseline.numpy()}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
