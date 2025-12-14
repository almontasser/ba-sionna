"""
End-to-End Beam Alignment Model

This module implements a complete deep learning-based beam alignment system for
mmWave communication, integrating three main components:
1. Base Station (BS) Controller: Manages transmit beamforming with learned codebook
2. User Equipment (UE) Controller: Adaptively selects receive beams using RNN
3. Channel Model: Generates standardized 3GPP TR 38.901 scenario channels (UMi/UMa/RMa)

System Architecture (C3-only):
    The beam alignment process operates in T+1 phases:

    Sensing Phase (t=0 to T-1):
        For each step t:
        1. BS transmits with beam f_t from learned codebook
        2. UE generates receive beam w_t based on history (via RNN)
        3. UE measures received signal: y_t = w_t^H H f_t + noise
        4. UE updates internal RNN state with (y_t, x_t)
        5. UE generates feedback m_t

    Final Beamforming Phase (t=T):
        1. UE sends final feedback m_FB to BS
        2. BS generates final beam f_T using FNN(m_FB)
        3. UE generates final beam w_T from RNN state
        4. Compute objective: maximize |w_T^H H f_T|^2

Mathematical Model:
    Beamforming Gain: G = |w^H H f|^2
    Objective: max_{f,w} 𝔼[G / ||H||_F^2]

References:
    "Deep Learning Based Adaptive Joint mmWave Beam Alignment"
    arXiv:2401.13587v1
"""

import tensorflow as tf
import numpy as np
import sys
import os

from config import Config

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel_model import SionnaScenarioChannelModel, SIONNA_AVAILABLE
from utils import (
    compute_beamforming_gain,
    compute_beamforming_gain_db,
    add_complex_noise,
)
from models.bs_controller import BSController
from models.ue_controller import UEController


class BeamAlignmentModel(tf.keras.Model):
    """
    Complete end-to-end beam alignment system.

    System flow:
    1. Generate mmWave channel H
    2. For each sensing step t=0 to T-1:
        a. BS selects beam f_t from codebook
        b. UE generates combining vector w_t using RNN
        c. Compute received signal: y_t = w_t^H H f_t + noise
        d. UE updates internal state based on y_t and beam index
    3. Final beam pair is selected based on T sensing steps
    4. Compute beamforming gain for final beam pair
    """

    def __init__(
        self,
        num_tx_antennas,
        num_rx_antennas,
        codebook_size=8,
        num_sensing_steps=8,
        rnn_hidden_size=128,
        rnn_type="GRU",
        num_feedback=4,
        start_beam_index=0,
        random_start=False,
        carrier_frequency=28e9,
        scenarios=None,
        o2i_model="low",
        enable_pathloss=False,
        enable_shadow_fading=False,
        distance_range_m=(10.0, 200.0),
        ue_speed_range=(0.0, 30.0),
        indoor_probability=0.0,
        ut_height_m=1.5,
        bs_height_umi_m=10.0,
        bs_height_uma_m=25.0,
        bs_height_rma_m=35.0,
        narrowband_method=None,
        narrowband_subcarrier=None,
        **kwargs,
    ):
        """
        Args:
            num_tx_antennas: Number of BS transmit antennas (NTX)
            num_rx_antennas: Number of UE receive antennas (NRX)
            codebook_size: BS codebook size (NCB)
            num_sensing_steps: Number of sensing steps (T)
            rnn_hidden_size: UE RNN hidden size
            rnn_type: UE RNN type ("GRU" or "LSTM")
            num_feedback: Number of feedback values (NFB)
            start_beam_index: Starting beam index (if not random)
            random_start: Whether to use random starting beam
            carrier_frequency: Carrier frequency in Hz (for Sionna TR 38.901 scenarios)
            scenarios: List of scenario names (e.g., ["UMi","UMa","RMa"])
            o2i_model: O2I model for UMi/UMa ("low" or "high")
            enable_pathloss: Enable pathloss in scenario channel
            enable_shadow_fading: Enable shadow fading in scenario channel
            distance_range_m: (min, max) UT-BS 2D distance range in meters
            ue_speed_range: (min, max) UE speed range in m/s
            indoor_probability: Probability a UT is indoor (affects O2I)
            ut_height_m: UT height in meters
            bs_height_umi_m: BS height for UMi in meters
            bs_height_uma_m: BS height for UMa in meters
            bs_height_rma_m: BS height for RMa in meters
            narrowband_method: Narrowband reduction method ("center", "subcarrier", "mean_cfr")
            narrowband_subcarrier: Subcarrier index if narrowband_method=="subcarrier"
        """
        super().__init__(**kwargs)

        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.codebook_size = codebook_size
        self.num_sensing_steps = num_sensing_steps
        self.start_beam_index = start_beam_index
        self.random_start = random_start

        # Create channel model - Sionna TR 38.901 scenarios
        if not SIONNA_AVAILABLE:
            raise ImportError("Sionna must be installed for channel generation.")
        print("  Using Sionna TR 38.901 scenario channel model")
        if narrowband_method is None:
            narrowband_method = getattr(Config, "NARROWBAND_METHOD", "center")
        if narrowband_subcarrier is None:
            narrowband_subcarrier = getattr(Config, "NARROWBAND_SUBCARRIER", None)
        self.channel_model = SionnaScenarioChannelModel(
            num_tx_antennas=num_tx_antennas,
            num_rx_antennas=num_rx_antennas,
            carrier_frequency=carrier_frequency,
            scenarios=scenarios,
            o2i_model=o2i_model,
            enable_pathloss=enable_pathloss,
            enable_shadow_fading=enable_shadow_fading,
            distance_range_m=distance_range_m,
            ue_speed_range=ue_speed_range,
            indoor_probability=indoor_probability,
            ut_height_m=ut_height_m,
            bs_height_umi_m=bs_height_umi_m,
            bs_height_uma_m=bs_height_uma_m,
            bs_height_rma_m=bs_height_rma_m,
            fft_size=Config.RESOURCE_GRID_FFT_SIZE,
            num_ofdm_symbols=Config.RESOURCE_GRID_NUM_OFDM_SYMBOLS,
            subcarrier_spacing=Config.RESOURCE_GRID_SUBCARRIER_SPACING,
            narrowband_method=narrowband_method,
            narrowband_subcarrier=narrowband_subcarrier,
        )

        self.bs_controller = BSController(
            num_antennas=num_tx_antennas,
            codebook_size=codebook_size,
            initialize_with_dft=True,
            trainable_codebook=True,  # C3: learnable codebook
        )

        self.ue_controller = UEController(
            num_antennas=num_rx_antennas,
            rnn_hidden_size=rnn_hidden_size,
            rnn_type=rnn_type,
            num_feedback=num_feedback,
            codebook_size=codebook_size,
        )

    def execute_beam_alignment(
        self,
        channels,
        noise_power,
        training=False,
        start_idx=None,
        measurement_ablation=None,
        shuffle_perm=None,
    ):
        """
        Execute the beam alignment process.

        Args:
            channels: Channel matrices, shape (batch, nrx, ntx)
            noise_power: Noise power for received signal
            training: Whether in training mode
            start_idx: Optional starting beam indices for the BS sweep, shape (batch,).
                If provided, overrides random/deterministic start configured on the model.
            measurement_ablation: Optional measurement ablation for analysis/debugging.
                One of:
                  - None/"none": no ablation (default)
                  - "zero": force y_t = 0 (UE gets no information)
                  - "noise_only": force y_t = w_t^H n_t (signal removed)
                  - "shuffle": permute y_t across batch (break H↔y association)
            shuffle_perm: Optional fixed permutation for "shuffle" ablation, shape (batch,).
                If None and measurement_ablation=="shuffle", a random permutation is used.

        Returns:
            Dictionary with:
                - final_tx_beams: Final transmit beams (batch, ntx)
                - final_rx_beams: Final receive beams (batch, nrx)
                - beamforming_gain: Beamforming gains (batch,)
                - received_signals: All received signals (batch, T)
                - beam_indices: BS beam indices (batch, T)
        """
        batch_size = tf.shape(channels)[0]
        T = self.num_sensing_steps

        # Determine starting beam index
        if start_idx is None:
            if self.random_start:
                start_idx = tf.random.uniform(
                    [batch_size], minval=0, maxval=self.codebook_size, dtype=tf.int32
                )
            else:
                start_idx = self.start_beam_index
        else:
            start_idx = tf.cast(start_idx, tf.int32)

        measurement_ablation = (
            "none" if measurement_ablation is None else str(measurement_ablation).lower()
        )
        if measurement_ablation == "shuffle":
            if shuffle_perm is None:
                # Deterministic in-batch permutation for fair, repeatable ablations
                # (also avoids relying on global RNG state).
                shuffle_perm = tf.range(batch_size - 1, -1, -1, dtype=tf.int32)
            else:
                shuffle_perm = tf.cast(shuffle_perm, tf.int32)
            tf.debugging.assert_greater_equal(shuffle_perm, 0)
            tf.debugging.assert_less(shuffle_perm, batch_size)

        # Get BS beam sequence
        tx_beams_sequence, beam_indices = self.bs_controller.get_beam_sequence(
            start_index=start_idx, num_steps=T, batch_size=batch_size
        )  # (batch, T, ntx), (batch, T)

        # Initialize UE state
        ue_state = self.ue_controller.get_initial_state(batch_size)

        # Lists to store received signals and UE beams
        received_signals_list = []
        rx_beams_list = []

        for t in range(T):
            # Get BS beam for this step
            f_t = tx_beams_sequence[:, t, :]  # (batch, ntx)
            x_t = beam_indices[:, t]  # (batch,)

            # UE generates combining vector based on current state
            # For t=0, we use a default beam (e.g., omnidirectional)
            if t == 0:
                # Initial beam: omnidirectional (all ones normalized)
                w_t = tf.ones([batch_size, self.num_rx_antennas], dtype=tf.complex64)
                norm_factor = tf.cast(
                    tf.sqrt(tf.cast(self.num_rx_antennas, tf.float32)), tf.complex64
                )
                w_t = w_t / norm_factor
            else:
                # Use previous received signal to generate current beam
                y_prev = received_signals_list[-1]
                x_prev = beam_indices[:, t - 1]
                w_t, _, ue_state = self.ue_controller.process_step(y_prev, x_prev, ue_state)

            # Compute received signal: y_t = w_t^H @ H @ f_t + noise
            # H @ f_t
            Hf = tf.linalg.matvec(channels, f_t)  # (batch, nrx)

            # w_t^H @ (H @ f_t)
            signal = tf.reduce_sum(tf.math.conj(w_t) * Hf, axis=-1)  # (batch,)

            # Add noise
            if measurement_ablation == "noise_only":
                y_t = add_complex_noise(tf.zeros_like(signal), noise_power)
            else:
                y_t = add_complex_noise(signal, noise_power)

            if measurement_ablation == "zero":
                y_t = tf.zeros_like(y_t)
            elif measurement_ablation == "shuffle":
                y_t = tf.gather(y_t, shuffle_perm, axis=0)

            received_signals_list.append(y_t)
            rx_beams_list.append(w_t)

        # Stack all received signals and beams
        received_signals = tf.stack(received_signals_list, axis=1)  # (batch, T)
        rx_beams_sequence = tf.stack(rx_beams_list, axis=1)  # (batch, T, nrx)

        # ==================================================================
        # Final Beamforming Step (t=T)
        # ==================================================================

        # UE processes the last sensing measurement y_{T-1} to produce:
        #   w_T = g1(y_{T-1}, x_{T-1}, h_{T-1})
        #   m_FB = g2(y_{T-1}, x_{T-1}, h_{T-1})
        y_last = received_signals_list[-1]
        x_last = beam_indices[:, T - 1]
        final_rx_beam, final_feedback, _ = self.ue_controller.process_step(
            y_last, x_last, ue_state
        )

        # BS generates final beam f_T = g3(m_FB)
        final_tx_beam = self.bs_controller.get_beam_from_feedback(final_feedback)

        # 3. Compute final beamforming gain
        # This is the objective function to maximize
        final_beamforming_gain = compute_beamforming_gain(
            channels, final_tx_beam, final_rx_beam
        )

        results = {
            "final_tx_beams": final_tx_beam,
            "final_rx_beams": final_rx_beam,
            "beamforming_gain": final_beamforming_gain,  # Used for loss
            "received_signals": received_signals,
            "beam_indices": beam_indices,
            "tx_beams_sequence": tx_beams_sequence,
            "rx_beams_sequence": rx_beams_sequence,
            "feedback": final_feedback,
        }

        return results

    def call(self, batch_size, snr_db=5.0, training=False):
        """
        Forward pass: generate channels and perform beam alignment.

        Args:
            batch_size: Number of samples
            snr_db: SNR per antenna in dB
            training: Training mode flag

        Returns:
            Dictionary with alignment results
        """
        # Generate channels
        channels = self.channel_model.generate_channel(batch_size)

        # Compute noise power from per-antenna SNR.
        # Paper Eq. (4): SNR_ANT = 1/sigma_n^2 (pilot power normalized to 1),
        # hence sigma_n^2 = 1/SNR_ANT.
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1.0 / snr_linear

        # Execute beam alignment
        results = self.execute_beam_alignment(channels, noise_power, training=training)

        # Add channel to results
        results["channels"] = channels
        results["snr_db"] = snr_db
        results["noise_power"] = noise_power

        return results


if __name__ == "__main__":
    print("Testing Beam Alignment Model...")
    print("=" * 60)

    # Create model
    model = BeamAlignmentModel(
        num_tx_antennas=64,
        num_rx_antennas=16,
        codebook_size=8,
        num_sensing_steps=8,
        rnn_hidden_size=128,
        rnn_type="GRU",
        num_feedback=4,
        start_beam_index=0,
        random_start=False,
    )

    # Test forward pass
    batch_size = 10
    snr_db = 5.0

    results = model(batch_size=batch_size, snr_db=snr_db, training=True)

    print(f"\nForward pass results:")
    print(f"  Channels shape: {results['channels'].shape}")
    print(f"  Final TX beams shape: {results['final_tx_beams'].shape}")
    print(f"  Final RX beams shape: {results['final_rx_beams'].shape}")
    print(f"  Beamforming gain shape: {results['beamforming_gain'].shape}")
    print(f"  Received signals shape: {results['received_signals'].shape}")
    print(f"  Beam indices shape: {results['beam_indices'].shape}")

    # Compute statistics
    bf_gain_db = 10 * tf.math.log(results["beamforming_gain"]) / tf.math.log(10.0)
    print(f"\nBeamforming gain statistics:")
    print(f"  Mean: {tf.reduce_mean(bf_gain_db):.2f} dB")
    print(f"  Std: {tf.math.reduce_std(bf_gain_db):.2f} dB")
    print(f"  Min: {tf.reduce_min(bf_gain_db):.2f} dB")
    print(f"  Max: {tf.reduce_max(bf_gain_db):.2f} dB")

    # Test trainability
    print(f"\nModel trainable variables:")
    print(f"  Total: {len(model.trainable_variables)}")
    for var in model.trainable_variables[:5]:
        print(f"    {var.name}: {var.shape}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
