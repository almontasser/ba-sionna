"""
Sionna-based 3GPP TR 38.901 CDL channel model with domain randomization.

This module solely uses Sionna's native CDL + OFDM pipeline to generate
frequency-flat channels for the beam-alignment model. No geometric fallback is
kept to ensure all training/evaluation rely on the standardized CDL models.
"""

import tensorflow as tf
try:
    import sionna
    # Prefer the newer sionna.phy API, fall back to legacy paths if needed.
    try:
        from sionna.phy.channel.tr38901 import CDL, PanelArray
        from sionna.phy.channel import GenerateOFDMChannel
        from sionna.phy.ofdm import ResourceGrid
    except Exception:  # noqa: BLE001
        # Legacy (<1.0) module structure
        from sionna.channel.tr38901 import CDL, PanelArray  # type: ignore
        from sionna.channel import GenerateOFDMChannel  # type: ignore
        from sionna.ofdm import ResourceGrid  # type: ignore
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("Error: Sionna not available. Install with `pip install sionna`.")


class SionnaCDLChannelModel(tf.keras.layers.Layer):
    """
    Sionna 3GPP TR 38.901 CDL Channel Model with Domain Randomization.
    
    This implementation now uses Sionna's native CDL + OFDM channel pipeline
    (ResourceGrid + GenerateOFDMChannel) instead of the previous manual
    parametric reconstruction. The output remains frequency-flat with shape
    (batch_size, num_rx_antennas, num_tx_antennas) to stay plug‑compatible with
    the rest of the beam-alignment code.
    
    Pipeline:
        CDL (TR38.901) --> GenerateOFDMChannel --> average over OFDM symbols &
        subcarriers --> H ∈ ℂ^{batch × NRX × NTX}
    
    Domain randomization:
        - Random CDL profile per batch (A/B/C/D/E)
        - Random delay spread per batch (uniform in delay_spread_range)
        - UE speed randomization is handled internally by Sionna via min/max speed
    """
    
    def __init__(self, 
                 num_tx_antennas, 
                 num_rx_antennas,
                 carrier_frequency=28e9,
                 delay_spread_range=(10e-9, 300e-9),  # 10ns to 300ns
                 ue_speed_range=(0.0, 30.0),  # 0 to 30 m/s (108 km/h)
                 cdl_models=None,
                 fft_size=64,
                 num_ofdm_symbols=1,
                 subcarrier_spacing=120e3,
                 **kwargs):
        """
        Args:
            num_tx_antennas: Number of BS transmit antennas (NTX)
            num_rx_antennas: Number of UE receive antennas (NRX)
            carrier_frequency: Carrier frequency in Hz (default: 28 GHz for mmWave)
            delay_spread_range: (min, max) delay spread in seconds for randomization
            ue_speed_range: (min, max) UE speed in m/s for Doppler randomization
            cdl_models: List of CDL model names (default: all 5 models)
                       Options: "A", "B", "C", "D", "E"
            fft_size: OFDM FFT size used when generating frequency responses
            num_ofdm_symbols: Number of OFDM symbols to generate (averaged)
            subcarrier_spacing: Subcarrier spacing in Hz
        """
        super().__init__(**kwargs)
        
        if not SIONNA_AVAILABLE:
            raise ImportError(
                "Sionna is not installed. Install with: pip install sionna\n"
                "See: https://nvlabs.github.io/sionna/"
            )
        
        self.num_tx_antennas = num_tx_antennas
        self.num_rx_antennas = num_rx_antennas
        self.carrier_frequency = carrier_frequency
        self.delay_spread_range = delay_spread_range
        self.ue_speed_range = ue_speed_range
        self.fft_size = fft_size
        self.num_ofdm_symbols = num_ofdm_symbols
        self.subcarrier_spacing = subcarrier_spacing
        
        # Default to all CDL models for maximum diversity
        if cdl_models is None:
            cdl_models = ["A", "B", "C", "D", "E"]
        self.cdl_models = cdl_models
        self.num_cdl_models = len(cdl_models)
        
        # Build antenna arrays (ULA) for BS and UE
        self.bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=num_tx_antennas,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency,
        )
        self.ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=num_rx_antennas,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )
        
        # Resource grid for OFDM channel generation
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=1,
        )
        
        # Pre-instantiate one CDL + GenerateOFDMChannel per profile
        self._cdl_models = []
        self._ofdm_channels = []
        for model_name in cdl_models:
            cdl = CDL(
                model=model_name,
                delay_spread=delay_spread_range[0],
                carrier_frequency=carrier_frequency,
                ut_array=self.ut_array,
                bs_array=self.bs_array,
                direction="downlink",
                min_speed=ue_speed_range[0],
                max_speed=ue_speed_range[1],
            )
            ofdm_channel = GenerateOFDMChannel(
                channel_model=cdl,
                resource_grid=self.resource_grid
            )
            self._cdl_models.append(cdl)
            self._ofdm_channels.append(ofdm_channel)
        
        self._subcarrier_axes = (-1, -2)  # (subcarriers, OFDM symbols)
        print(f"✓ Sionna CDL Channel Model initialized (native)")
        print(f"  - CDL profiles: {', '.join(['CDL-' + m for m in cdl_models])}")
        print(f"  - Carrier frequency: {carrier_frequency/1e9:.1f} GHz")
        print(f"  - Delay spread range: {delay_spread_range[0]*1e9:.0f}-{delay_spread_range[1]*1e9:.0f} ns")
        print(f"  - UE speed range: {ue_speed_range[0]:.1f}-{ue_speed_range[1]:.1f} m/s")
        print(f"  - OFDM grid: fft_size={fft_size}, subcarrier_spacing={subcarrier_spacing/1e3:.0f} kHz")
        print(f"  - Antennas: {num_tx_antennas} BS, {num_rx_antennas} UE")
    
    def generate_channel(self, batch_size):
        """
        Generate a batch of CDL channel realizations with domain randomization.
        
        This method randomly samples:
        - CDL profile (A/B/C/D/E) for each batch element
        - UE speed (affects Doppler shift) - Currently not used (quasi-static)
        - Delay spread (affects multipath severity)
        
        The resulting channels have diverse characteristics that train robust models.
        
        Args:
            batch_size: Number of channel samples to generate
            
        Returns:
            Channel tensor of shape (batch_size, num_rx_antennas, num_tx_antennas)
            
        Note:
            This implementation leverages Sionna's CDL + OFDM pipeline and then
            averages over subcarriers and OFDM symbols to yield a frequency-flat
            channel matrix compatible with the rest of the codebase.
        """
        # Randomly select CDL model index
        cdl_idx = tf.random.uniform(
            [], minval=0, maxval=self.num_cdl_models, dtype=tf.int32
        )

        # Random delay spread for this batch
        delay_spread = tf.random.uniform(
            [], self.delay_spread_range[0], self.delay_spread_range[1], tf.float32
        )

        def _gen_with_model(model_idx):
            # Assign delay spread (Sionna samples speeds internally)
            self._cdl_models[model_idx].delay_spread = delay_spread
            h_freq = self._ofdm_channels[model_idx](batch_size)
            # Average over subcarriers and OFDM symbols to get flat channel
            h_flat = tf.reduce_mean(h_freq, axis=self._subcarrier_axes)
            # Remove singleton num_rx and num_tx dims: (batch, nrx_ant, ntx_ant)
            h_flat = tf.squeeze(h_flat, axis=[1, 3])
            return tf.cast(h_flat, tf.complex64)

        if self.num_cdl_models == 1:
            return _gen_with_model(0)

        branch_fns = {i: (lambda i=i: _gen_with_model(i)) for i in range(self.num_cdl_models)}
        return tf.switch_case(cdl_idx, branch_fns=branch_fns)
    
    def call(self, batch_size):
        """
        Call method for Keras Layer API.
        
        Args:
            batch_size: Number of channels to generate (can be a tensor)
            
        Returns:
            Channel tensor
        """
        if isinstance(batch_size, tf.Tensor):
            batch_size = tf.get_static_value(batch_size)
            if batch_size is None:
                raise ValueError("batch_size must be known at graph construction time")
        
        return self.generate_channel(batch_size)


if __name__ == "__main__":
    print("Testing Sionna CDL Channel Model...")
    channel_model = SionnaCDLChannelModel(
        num_tx_antennas=32,
        num_rx_antennas=16,
        carrier_frequency=28e9,
        cdl_models=["A"],
        delay_spread_range=(30e-9, 30e-9),
        ue_speed_range=(0.0, 0.0),
    )
    H = channel_model.generate_channel(batch_size=4)
    print(f"Channel tensor shape: {H.shape}")
    print(f"Channel dtype: {H.dtype}")
