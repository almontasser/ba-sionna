"""
Baselines for beam alignment evaluation.

Currently implemented:
  - ExhaustiveSearchBaseline: brute-force best reminds (tx, rx) beam pair from codebooks.
"""

from __future__ import annotations

import tensorflow as tf

from utils import compute_beamforming_gain


class ExhaustiveSearchBaseline:
    """
    Exhaustive search baseline for comparison.

    Tests all possible beam pairs to find the optimal one under:
      gain(tx,rx) = |w^H H f|^2
    """

    def __init__(self, tx_codebook: tf.Tensor, rx_codebook: tf.Tensor):
        """
        Args:
            tx_codebook: Transmit beam codebook (NCB_tx, ntx)
            rx_codebook: Receive beam codebook (NCB_rx, nrx)
        """
        self.tx_codebook = tx_codebook
        self.rx_codebook = rx_codebook
        self.ncb_tx = int(tx_codebook.shape[0])
        self.ncb_rx = int(rx_codebook.shape[0])

    def find_best_beam_pair(self, channel: tf.Tensor):
        """
        Find best beam pair via exhaustive search for a single channel.

        Args:
            channel: Channel matrix (nrx, ntx)

        Returns:
            Best transmit beam index, best receive beam index, best gain (linear)
        """
        best_gain = -float("inf")
        best_tx_idx = 0
        best_rx_idx = 0

        for tx_idx in range(self.ncb_tx):
            for rx_idx in range(self.ncb_rx):
                tx_beam = self.tx_codebook[tx_idx]
                rx_beam = self.rx_codebook[rx_idx]

                gain = compute_beamforming_gain(
                    tf.expand_dims(channel, 0),
                    tf.expand_dims(tx_beam, 0),
                    tf.expand_dims(rx_beam, 0),
                )[0]

                if gain > best_gain:
                    best_gain = gain
                    best_tx_idx = tx_idx
                    best_rx_idx = rx_idx

        return best_tx_idx, best_rx_idx, best_gain

    def search_batch(self, channels: tf.Tensor):
        """
        Exhaustive search for a batch of channels.

        Args:
            channels: Channel matrices (batch, nrx, ntx)

        Returns:
            Dictionary with best beams and gains
        """
        batch_size = int(channels.shape[0])

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
            "tx_beams": best_tx_beams,
            "rx_beams": best_rx_beams,
            "bf_gains": tf.stack(best_gains),
        }

