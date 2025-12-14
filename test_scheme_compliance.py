#!/usr/bin/env python3
"""
Smoke/Compliance Test Suite (C3-only)

This repo was simplified to a single end-to-end beam-alignment pipeline
corresponding to the paper's "C3" variant:
  - UE RNN controller (N1)
  - BS feedback network (N2)
  - Learnable BS codebook (N3)

Tests:
1) Forward pass shape checks
2) Gradient flow through all trainable variables
3) Codebook parameters update under optimization
"""

import numpy as np
import tensorflow as tf

from config import Config
from device_setup import setup_device
from models.beam_alignment import BeamAlignmentModel


def test_forward_shapes():
    print("\n=== TEST 1: Forward pass shapes ===")
    model = BeamAlignmentModel(
        num_tx_antennas=Config.NTX,
        num_rx_antennas=Config.NRX,
        codebook_size=Config.NCB,
        num_sensing_steps=Config.T,
        rnn_hidden_size=Config.RNN_HIDDEN_SIZE,
        rnn_type=Config.RNN_TYPE,
        num_feedback=Config.NUM_FEEDBACK,
        start_beam_index=Config.START_BEAM_INDEX,
        random_start=Config.RANDOM_START,
        carrier_frequency=Config.CARRIER_FREQUENCY,
        scenarios=Config.SCENARIOS,
        o2i_model=getattr(Config, "O2I_MODEL", "low"),
        enable_pathloss=getattr(Config, "ENABLE_PATHLOSS", False),
        enable_shadow_fading=getattr(Config, "ENABLE_SHADOW_FADING", False),
        distance_range_m=getattr(Config, "DISTANCE_RANGE_M", (10.0, 200.0)),
        ue_speed_range=getattr(Config, "UE_SPEED_RANGE", (0.0, 30.0)),
        indoor_probability=getattr(Config, "INDOOR_PROBABILITY", 0.0),
        ut_height_m=getattr(Config, "UT_HEIGHT_M", 1.5),
        bs_height_umi_m=getattr(Config, "BS_HEIGHT_UMI_M", 10.0),
        bs_height_uma_m=getattr(Config, "BS_HEIGHT_UMA_M", 25.0),
        bs_height_rma_m=getattr(Config, "BS_HEIGHT_RMA_M", 35.0),
    )

    results = model(batch_size=4, snr_db=5.0, training=False)

    assert results["channels"].shape == (4, Config.NRX, Config.NTX)
    assert results["final_tx_beams"].shape == (4, Config.NTX)
    assert results["final_rx_beams"].shape == (4, Config.NRX)
    assert results["beamforming_gain"].shape == (4,)
    assert results["received_signals"].shape == (4, Config.T)
    assert results["beam_indices"].shape == (4, Config.T)
    print("✓ Shapes OK")


def test_gradient_flow():
    print("\n=== TEST 2: Gradient flow ===")
    model = BeamAlignmentModel(
        num_tx_antennas=Config.NTX,
        num_rx_antennas=Config.NRX,
        codebook_size=Config.NCB,
        num_sensing_steps=Config.T,
        rnn_hidden_size=Config.RNN_HIDDEN_SIZE,
        rnn_type=Config.RNN_TYPE,
        num_feedback=Config.NUM_FEEDBACK,
        start_beam_index=Config.START_BEAM_INDEX,
        random_start=Config.RANDOM_START,
        carrier_frequency=Config.CARRIER_FREQUENCY,
        scenarios=Config.SCENARIOS,
        o2i_model=getattr(Config, "O2I_MODEL", "low"),
        enable_pathloss=getattr(Config, "ENABLE_PATHLOSS", False),
        enable_shadow_fading=getattr(Config, "ENABLE_SHADOW_FADING", False),
        distance_range_m=getattr(Config, "DISTANCE_RANGE_M", (10.0, 200.0)),
        ue_speed_range=getattr(Config, "UE_SPEED_RANGE", (0.0, 30.0)),
        indoor_probability=getattr(Config, "INDOOR_PROBABILITY", 0.0),
        ut_height_m=getattr(Config, "UT_HEIGHT_M", 1.5),
        bs_height_umi_m=getattr(Config, "BS_HEIGHT_UMI_M", 10.0),
        bs_height_uma_m=getattr(Config, "BS_HEIGHT_UMA_M", 25.0),
        bs_height_rma_m=getattr(Config, "BS_HEIGHT_RMA_M", 35.0),
    )

    with tf.GradientTape() as tape:
        results = model(batch_size=8, snr_db=5.0, training=True)
        loss = -tf.reduce_mean(results["beamforming_gain"])

    gradients = tape.gradient(loss, model.trainable_variables)
    missing = [v.name for v, g in zip(model.trainable_variables, gradients) if g is None]
    assert not missing, f"Missing gradients for: {missing}"
    print(f"✓ Gradients OK ({len(model.trainable_variables)}/{len(model.trainable_variables)})")


def test_codebook_updates():
    print("\n=== TEST 3: Codebook updates ===")
    model = BeamAlignmentModel(
        num_tx_antennas=Config.NTX,
        num_rx_antennas=Config.NRX,
        codebook_size=Config.NCB,
        num_sensing_steps=Config.T,
        rnn_hidden_size=Config.RNN_HIDDEN_SIZE,
        rnn_type=Config.RNN_TYPE,
        num_feedback=Config.NUM_FEEDBACK,
        start_beam_index=Config.START_BEAM_INDEX,
        random_start=Config.RANDOM_START,
        carrier_frequency=Config.CARRIER_FREQUENCY,
        scenarios=Config.SCENARIOS,
        o2i_model=getattr(Config, "O2I_MODEL", "low"),
        enable_pathloss=getattr(Config, "ENABLE_PATHLOSS", False),
        enable_shadow_fading=getattr(Config, "ENABLE_SHADOW_FADING", False),
        distance_range_m=getattr(Config, "DISTANCE_RANGE_M", (10.0, 200.0)),
        ue_speed_range=getattr(Config, "UE_SPEED_RANGE", (0.0, 30.0)),
        indoor_probability=getattr(Config, "INDOOR_PROBABILITY", 0.0),
        ut_height_m=getattr(Config, "UT_HEIGHT_M", 1.5),
        bs_height_umi_m=getattr(Config, "BS_HEIGHT_UMI_M", 10.0),
        bs_height_uma_m=getattr(Config, "BS_HEIGHT_UMA_M", 25.0),
        bs_height_rma_m=getattr(Config, "BS_HEIGHT_RMA_M", 35.0),
    )

    # Build
    _ = model(batch_size=2, snr_db=5.0, training=True)

    codebook_vars = [v for v in model.trainable_variables if "codebook_" in v.name]
    assert len(codebook_vars) == 2, f"Expected 2 codebook vars, got {len(codebook_vars)}"

    initial = [v.numpy().copy() for v in codebook_vars]
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    for _ in range(3):
        with tf.GradientTape() as tape:
            results = model(batch_size=8, snr_db=5.0, training=True)
            loss = -tf.reduce_mean(results["beamforming_gain"])
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    final = [v.numpy() for v in codebook_vars]
    changed = any(np.any(np.abs(f - i) > 1e-7) for f, i in zip(final, initial))
    assert changed, "Codebook variables did not change under optimization"
    print("✓ Codebook updates OK")


def main():
    print("🔍 Beam Alignment C3-only Test Suite")
    print("=" * 60)
    setup_device(verbose=False)
    test_forward_shapes()
    test_gradient_flow()
    test_codebook_updates()
    print("\n✅ All tests passed.")


if __name__ == "__main__":
    main()
