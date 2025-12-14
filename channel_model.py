"""
Compatibility wrapper for channel models.

The implementation lives in `channels/tr38901_scenario.py`, but many scripts
import from `channel_model.py`. This module re-exports the public symbols.
"""

from channels.tr38901_scenario import SIONNA_AVAILABLE, SionnaScenarioChannelModel

__all__ = ["SIONNA_AVAILABLE", "SionnaScenarioChannelModel"]


if __name__ == "__main__":
    print("Testing Sionna Scenario Channel Model...")
    channel_model = SionnaScenarioChannelModel(
        num_tx_antennas=32,
        num_rx_antennas=16,
        carrier_frequency=28e9,
        scenarios=["UMi", "UMa", "RMa"],
        enable_pathloss=False,
        enable_shadow_fading=False,
        indoor_probability=0.0,
        narrowband_method="center",
    )
    H = channel_model.generate_channel(batch_size=4)
    print(f"Channel tensor shape: {H.shape}")
    print(f"Channel dtype: {H.dtype}")
