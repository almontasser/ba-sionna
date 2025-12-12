"""
Sionna-based 3GPP TR 38.901 CDL channel model with domain randomization.

This module solely uses Sionna's native CDL + OFDM pipeline to generate
frequency-flat channels for the beam-alignment model. No geometric fallback is
kept to ensure all training/evaluation rely on the standardized CDL models.
"""

import tensorflow as tf
try:
    import sionna  # noqa: F401
    # Prefer the newer sionna.phy API, fall back to legacy paths if needed.
    try:
        from sionna.phy.channel.tr38901 import CDL, PanelArray
    except Exception:  # noqa: BLE001
        from sionna.channel.tr38901 import CDL, PanelArray  # type: ignore
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("Error: Sionna not available. Install with `pip install sionna`.")


class SionnaCDLChannelModel(tf.keras.layers.Layer):
    """
    Sionna 3GPP TR 38.901 CDL Channel Model with Domain Randomization.
    
    This implementation uses Sionna's native CDL CIR sampler and then maps the
    resulting frequency-selective channel to a narrowband H used by the beam-
    alignment model.

    Pipeline:
        CDL (TR38.901) --> CIR (h, tau) --> narrowband mapping --> H ∈ ℂ^{batch × NRX × NTX}

    Domain randomization (per *sample*):
        - Random CDL profile per batch element (A/B/C/D/E)
        - Random delay spread per batch element (uniform in delay_spread_range)
        - UE speed randomization is handled internally by Sionna via min/max speed

    Narrowband mapping options:
        - "center": DC/center-subcarrier (paper-consistent flat fading)
        - "subcarrier": pick a specific subcarrier index
        - "mean_cfr": average complex CFR over all subcarriers
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
                 narrowband_method="center",
                 narrowband_subcarrier=None,
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
            num_ofdm_symbols: Number of OFDM symbols to generate (only 1 is used for quasi-static sensing)
            subcarrier_spacing: Subcarrier spacing in Hz
            narrowband_method: Narrowband reduction method ("center", "subcarrier", "mean_cfr")
            narrowband_subcarrier: Subcarrier index if narrowband_method=="subcarrier"
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
        self.fft_size = int(fft_size)
        self.num_ofdm_symbols = int(num_ofdm_symbols)
        self.subcarrier_spacing = float(subcarrier_spacing)
        self.narrowband_method = narrowband_method
        self.narrowband_subcarrier = narrowband_subcarrier
        
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
        
        # Pre-instantiate one CDL CIR sampler per profile
        self._cdl_models = []
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
            self._cdl_models.append(cdl)

        # Base delay spread used for CIR sampling (scalar required by Sionna).
        # We rescale delays per sample afterwards.
        self._base_delay_spread = float(delay_spread_range[0])
        if self._base_delay_spread <= 0.0:
            raise ValueError("delay_spread_range[0] must be > 0 for rescaling.")

        print(f"✓ Sionna CDL Channel Model initialized (native)")
        print(f"  - CDL profiles: {', '.join(['CDL-' + m for m in cdl_models])}")
        print(f"  - Carrier frequency: {carrier_frequency/1e9:.1f} GHz")
        print(f"  - Delay spread range: {delay_spread_range[0]*1e9:.0f}-{delay_spread_range[1]*1e9:.0f} ns")
        print(f"  - UE speed range: {ue_speed_range[0]:.1f}-{ue_speed_range[1]:.1f} m/s")
        print(f"  - OFDM grid: fft_size={fft_size}, subcarrier_spacing={subcarrier_spacing/1e3:.0f} kHz")
        print(
            f"  - Narrowband method: {narrowband_method}"
            + (
                f" (k={narrowband_subcarrier})"
                if narrowband_method == "subcarrier"
                else ""
            )
        )
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
        batch_size = int(batch_size)

        # Sample CDL profile + delay spread per sample
        cdl_idx = tf.random.uniform(
            [batch_size], minval=0, maxval=self.num_cdl_models, dtype=tf.int32
        )
        delay_spread = tf.random.uniform(
            [batch_size], self.delay_spread_range[0], self.delay_spread_range[1], tf.float32
        )

        # Output tensor
        h_out = tf.zeros(
            [batch_size, self.num_rx_antennas, self.num_tx_antennas],
            dtype=tf.complex64,
        )

        # Precompute subcarrier frequency offsets (relative to DC)
        if self.narrowband_method == "center":
            f_offsets = tf.zeros([1], tf.float32)  # DC only
        elif self.narrowband_method == "subcarrier":
            k = self.narrowband_subcarrier
            if k is None:
                k = self.fft_size // 2
            f0 = (float(k) - (self.fft_size / 2.0)) * self.subcarrier_spacing
            f_offsets = tf.constant([f0], tf.float32)
        elif self.narrowband_method == "mean_cfr":
            k = tf.range(self.fft_size, dtype=tf.float32)
            f_offsets = (k - (self.fft_size / 2.0)) * self.subcarrier_spacing  # (K,)
        else:
            raise ValueError(
                f"Unknown narrowband_method '{self.narrowband_method}'. "
                "Use 'center', 'subcarrier', or 'mean_cfr'."
            )

        two_pi = tf.constant(2.0 * 3.141592653589793, tf.float32)

        # Generate per-profile CIR blocks, then scatter into output
        for mi in range(self.num_cdl_models):
            idxs = tf.where(tf.equal(cdl_idx, mi))[:, 0]
            num_i = tf.shape(idxs)[0]

            if tf.get_static_value(num_i) == 0:
                continue

            def _gen_block():
                ds_i = tf.gather(delay_spread, idxs)  # (num_i,)

                # CIR sampling requires scalar delay spread; use base then rescale delays per sample
                self._cdl_models[mi].delay_spread = self._base_delay_spread
                h_cir, tau = self._cdl_models[mi](
                    batch_size=num_i, num_time_steps=1, sampling_frequency=1.0
                )

                # Squeeze singleton dims
                # h_cir: (num_i, 1, nrx, 1, ntx, P, 1) -> (num_i, nrx, ntx, P)
                h_cir_s = tf.squeeze(h_cir, axis=[1, 3, 6])
                # tau: (num_i, 1, 1, P) -> (num_i, P)
                tau_s = tf.squeeze(tau, axis=[1, 2])

                # Normalize and rescale delays per sample
                tau_norm = tau_s / self._base_delay_spread  # (num_i, P)
                tau_scaled = tau_norm * tf.expand_dims(ds_i, axis=-1)  # (num_i, P)

                # Phase term: exp(-j 2π f_k τ_p)
                phase = -two_pi * tf.expand_dims(tau_scaled, axis=-1) * tf.reshape(
                    f_offsets, [1, 1, -1]
                )  # (num_i, P, K)
                exp_term = tf.exp(tf.complex(tf.zeros_like(phase), phase))  # (num_i, P, K)

                # Frequency response per subcarrier: sum_p h_p * exp(...)
                # h_cir_s: (num_i, nrx, ntx, P), exp_term: (num_i, P, K)
                h_freq = tf.einsum("b i j p, b p k -> b i j k", h_cir_s, exp_term)

                if self.narrowband_method == "mean_cfr":
                    h_flat = tf.reduce_mean(h_freq, axis=-1)  # (num_i, nrx, ntx)
                else:
                    h_flat = tf.squeeze(h_freq, axis=-1)  # (num_i, nrx, ntx)

                return tf.cast(h_flat, tf.complex64)

            h_block = tf.cond(
                num_i > 0,
                true_fn=_gen_block,
                false_fn=lambda: tf.zeros([0, self.num_rx_antennas, self.num_tx_antennas], tf.complex64),
            )

            h_out = tf.tensor_scatter_nd_update(
                h_out,
                tf.expand_dims(idxs, axis=1),
                h_block,
            )

        return h_out
    
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
