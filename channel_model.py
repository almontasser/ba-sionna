"""
Sionna-based 3GPP TR 38.901 channel model wrappers.

Implemented:
  - SionnaScenarioChannelModel: TR 38.901 stochastic scenario models
    (UMi/UMa/RMa), mapped to a narrowband channel matrix H for the beam-alignment
    measurement model y_t = w_t^H H f_t + w_t^H n_t.
"""

import tensorflow as tf

try:
    import sionna  # noqa: F401

    # Prefer the newer sionna.phy API, fall back to legacy paths if needed.
    try:
        from sionna.phy.channel.tr38901 import PanelArray, UMi, UMa, RMa
    except Exception:  # noqa: BLE001
        from sionna.channel.tr38901 import PanelArray, UMi, UMa, RMa  # type: ignore

    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("Error: Sionna not available. Install with `pip install sionna`.")


def _frequency_offsets(narrowband_method, fft_size, subcarrier_spacing, narrowband_subcarrier):
    """Return subcarrier frequency offsets relative to DC (Hz)."""
    if narrowband_method == "center":
        return tf.zeros([1], tf.float32)  # DC only

    if narrowband_method == "subcarrier":
        k = narrowband_subcarrier
        if k is None:
            k = int(fft_size) // 2
        f0 = (float(k) - (float(fft_size) / 2.0)) * float(subcarrier_spacing)
        return tf.constant([f0], tf.float32)

    if narrowband_method == "mean_cfr":
        k = tf.range(int(fft_size), dtype=tf.float32)
        return (k - (float(fft_size) / 2.0)) * float(subcarrier_spacing)  # (K,)

    raise ValueError(
        f"Unknown narrowband_method '{narrowband_method}'. "
        "Use 'center', 'subcarrier', or 'mean_cfr'."
    )


def _cir_to_cfr(h_cir, tau, f_offsets_hz):
    """
    Convert CIR to CFR at specified frequency offsets.

    Args:
        h_cir: (B, NRX, NTX, P) complex path coefficients
        tau: (B, P) delays in seconds
        f_offsets_hz: (K,) frequency offsets relative to DC in Hz

    Returns:
        h_freq: (B, NRX, NTX, K)
    """
    two_pi = tf.constant(2.0 * 3.141592653589793, tf.float32)
    phase = -two_pi * tf.expand_dims(tau, axis=-1) * tf.reshape(
        f_offsets_hz, [1, 1, -1]
    )  # (B, P, K)
    exp_term = tf.exp(tf.complex(tf.zeros_like(phase), phase))  # (B, P, K)
    return tf.einsum("b i j p, b p k -> b i j k", h_cir, exp_term)


def _reduce_to_narrowband(h_freq, narrowband_method):
    """Reduce CFR to a narrowband H with shape (B, NRX, NTX)."""
    if narrowband_method == "mean_cfr":
        return tf.reduce_mean(h_freq, axis=-1)
    return tf.squeeze(h_freq, axis=-1)


class SionnaScenarioChannelModel(tf.keras.layers.Layer):
    """
    TR 38.901 stochastic scenario channel (UMi/UMa/RMa), mapped to narrowband H.

    This uses Sionna's system-level channel models, which sample the path angles
    and delays based on a sampled topology (BS/UT locations, heights, mobility,
    indoor/outdoor state). Compared to CDL profiles, this avoids fixed, profile-
    constant AoD/AoA values.

    Public interface:
      - generate_channel(batch_size) -> (B, NRX, NTX) complex64
    """

    def __init__(
        self,
        num_tx_antennas,
        num_rx_antennas,
        carrier_frequency=28e9,
        scenarios=None,
        o2i_model="low",  # required for UMi/UMa ("low" or "high")
        enable_pathloss=False,
        enable_shadow_fading=False,
        distance_range_m=(10.0, 200.0),
        ue_speed_range=(0.0, 30.0),
        indoor_probability=0.0,
        ut_height_m=1.5,
        bs_height_umi_m=10.0,
        bs_height_uma_m=25.0,
        bs_height_rma_m=35.0,
        fft_size=64,
        num_ofdm_symbols=1,
        subcarrier_spacing=120e3,
        narrowband_method="center",
        narrowband_subcarrier=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not SIONNA_AVAILABLE:
            raise ImportError(
                "Sionna is not installed. Install with: pip install sionna\n"
                "See: https://nvlabs.github.io/sionna/"
            )

        self.num_tx_antennas = int(num_tx_antennas)
        self.num_rx_antennas = int(num_rx_antennas)
        self.carrier_frequency = float(carrier_frequency)
        if scenarios is None:
            scenarios = ["UMi", "UMa", "RMa"]
        self.scenarios = [str(s) for s in scenarios]
        self.num_scenarios = len(self.scenarios)

        self.o2i_model = str(o2i_model)
        self.enable_pathloss = bool(enable_pathloss)
        self.enable_shadow_fading = bool(enable_shadow_fading)
        self.distance_range_m = (float(distance_range_m[0]), float(distance_range_m[1]))
        self.ue_speed_range = (float(ue_speed_range[0]), float(ue_speed_range[1]))
        self.indoor_probability = float(indoor_probability)
        self.ut_height_m = float(ut_height_m)
        self.bs_height_umi_m = float(bs_height_umi_m)
        self.bs_height_uma_m = float(bs_height_uma_m)
        self.bs_height_rma_m = float(bs_height_rma_m)

        self.fft_size = int(fft_size)
        self.num_ofdm_symbols = int(num_ofdm_symbols)  # kept for config compatibility
        self.subcarrier_spacing = float(subcarrier_spacing)
        self.narrowband_method = str(narrowband_method)
        self.narrowband_subcarrier = narrowband_subcarrier

        if not (0.0 <= self.indoor_probability <= 1.0):
            raise ValueError("indoor_probability must be in [0,1].")

        d_min, d_max = self.distance_range_m
        if d_min <= 0.0 or d_max <= d_min:
            raise ValueError("distance_range_m must satisfy 0 < min < max.")

        # Antenna arrays (ULA via single-panel array)
        self.bs_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=self.num_tx_antennas,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=self.carrier_frequency,
        )
        self.ut_array = PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=self.num_rx_antennas,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency,
        )

        # Pre-instantiate one system-level channel per scenario
        self._scenario_models = []
        for s in self.scenarios:
            if s == "UMi":
                self._scenario_models.append(
                    UMi(
                        carrier_frequency=self.carrier_frequency,
                        o2i_model=self.o2i_model,
                        ut_array=self.ut_array,
                        bs_array=self.bs_array,
                        direction="downlink",
                        enable_pathloss=self.enable_pathloss,
                        enable_shadow_fading=self.enable_shadow_fading,
                    )
                )
            elif s == "UMa":
                self._scenario_models.append(
                    UMa(
                        carrier_frequency=self.carrier_frequency,
                        o2i_model=self.o2i_model,
                        ut_array=self.ut_array,
                        bs_array=self.bs_array,
                        direction="downlink",
                        enable_pathloss=self.enable_pathloss,
                        enable_shadow_fading=self.enable_shadow_fading,
                    )
                )
            elif s == "RMa":
                self._scenario_models.append(
                    RMa(
                        carrier_frequency=self.carrier_frequency,
                        ut_array=self.ut_array,
                        bs_array=self.bs_array,
                        direction="downlink",
                        enable_pathloss=self.enable_pathloss,
                        enable_shadow_fading=self.enable_shadow_fading,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown scenario '{s}'. Expected one of: UMi, UMa, RMa."
                )

        self._f_offsets = _frequency_offsets(
            self.narrowband_method,
            self.fft_size,
            self.subcarrier_spacing,
            self.narrowband_subcarrier,
        )

        print("✓ Sionna Scenario Channel Model initialized (TR 38.901)")
        print(f"  - Scenarios: {', '.join(self.scenarios)}")
        print(f"  - Carrier frequency: {self.carrier_frequency/1e9:.1f} GHz")
        print(
            f"  - Distance range: {self.distance_range_m[0]:.1f}-{self.distance_range_m[1]:.1f} m"
        )
        print(
            f"  - UE speed range: {self.ue_speed_range[0]:.1f}-{self.ue_speed_range[1]:.1f} m/s"
        )
        print(
            f"  - Indoor probability: {self.indoor_probability:.2f} (used for UMi/UMa O2I)"
        )
        print(
            f"  - OFDM grid: fft_size={self.fft_size}, subcarrier_spacing={self.subcarrier_spacing/1e3:.0f} kHz"
        )
        nb_desc = self.narrowband_method
        if self.narrowband_method == "subcarrier":
            nb_desc += f" (k={self.narrowband_subcarrier})"
        print(f"  - Narrowband method: {nb_desc}")
        print(f"  - Antennas: {self.num_tx_antennas} BS, {self.num_rx_antennas} UE")

    def _bs_height_for_scenario(self, scenario):
        if scenario == "UMi":
            return self.bs_height_umi_m
        if scenario == "UMa":
            return self.bs_height_uma_m
        return self.bs_height_rma_m

    def _sample_topology(self, num_samples, scenario):
        """Sample a (num_samples,1,*) topology for one UT and one BS."""
        num_samples = tf.cast(num_samples, tf.int32)
        d_min, d_max = self.distance_range_m
        d_min = tf.constant(d_min, tf.float32)
        d_max = tf.constant(d_max, tf.float32)

        # Uniform-in-area radius sampling in [d_min, d_max]
        u = tf.random.uniform([num_samples], 0.0, 1.0, dtype=tf.float32)
        # Avoid tf.pow on GPU (can trigger XLA/libdevice issues in some installs).
        d_max_sq = d_max * d_max
        d_min_sq = d_min * d_min
        r = tf.sqrt(u * (d_max_sq - d_min_sq) + d_min_sq)
        phi = tf.random.uniform([num_samples], 0.0, 2.0 * 3.141592653589793, tf.float32)

        x = r * tf.cos(phi)
        y = r * tf.sin(phi)

        bs_h = tf.constant(self._bs_height_for_scenario(scenario), tf.float32)
        ut_h = tf.constant(self.ut_height_m, tf.float32)

        bs_loc = tf.stack([tf.zeros_like(x), tf.zeros_like(y), tf.fill([num_samples], bs_h)], axis=-1)
        ut_loc = tf.stack([x, y, tf.fill([num_samples], ut_h)], axis=-1)

        bs_loc = tf.expand_dims(bs_loc, axis=1)  # (B,1,3)
        ut_loc = tf.expand_dims(ut_loc, axis=1)  # (B,1,3)

        # Orientations: [yaw, pitch, roll] in radians
        bs_orient = tf.zeros([num_samples, 1, 3], tf.float32)
        ut_orient = tf.zeros([num_samples, 1, 3], tf.float32)

        # Velocities: random horizontal motion
        v_min, v_max = self.ue_speed_range
        speed = tf.random.uniform([num_samples], v_min, v_max, dtype=tf.float32)
        v_ang = tf.random.uniform([num_samples], 0.0, 2.0 * 3.141592653589793, tf.float32)
        vx = speed * tf.cos(v_ang)
        vy = speed * tf.sin(v_ang)
        vz = tf.zeros_like(vx)
        ut_vel = tf.stack([vx, vy, vz], axis=-1)
        ut_vel = tf.expand_dims(ut_vel, axis=1)  # (B,1,3)

        # Indoor/outdoor state (required by Sionna scenario models)
        if self.indoor_probability <= 0.0:
            in_state = tf.zeros([num_samples, 1], dtype=tf.bool)
        elif self.indoor_probability >= 1.0:
            in_state = tf.ones([num_samples, 1], dtype=tf.bool)
        else:
            in_state = (
                tf.random.uniform([num_samples, 1], 0.0, 1.0, dtype=tf.float32)
                < tf.constant(self.indoor_probability, tf.float32)
            )

        return ut_loc, bs_loc, ut_orient, bs_orient, ut_vel, in_state

    def _generate_scenario_block(self, scenario_idx, sample_indices):
        scenario = self.scenarios[scenario_idx]
        num_i = tf.shape(sample_indices)[0]

        sl = self._scenario_models[scenario_idx]
        ut_loc, bs_loc, ut_or, bs_or, ut_vel, in_state = self._sample_topology(num_i, scenario)
        sl.set_topology(
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orientations=ut_or,
            bs_orientations=bs_or,
            ut_velocities=ut_vel,
            in_state=in_state,
        )

        h_cir, tau = sl(num_time_samples=1, sampling_frequency=1.0)

        # Squeeze singleton dims:
        #   h_cir: (num_i, 1, nrx, 1, ntx, P, 1) -> (num_i, nrx, ntx, P)
        #   tau  : (num_i, 1, 1, P)            -> (num_i, P)
        h_cir_s = tf.squeeze(h_cir, axis=[1, 3, 6])
        tau_s = tf.squeeze(tau, axis=[1, 2])

        # CIR -> CFR -> narrowband H
        h_freq = _cir_to_cfr(h_cir_s, tau_s, self._f_offsets)
        h_flat = _reduce_to_narrowband(h_freq, self.narrowband_method)
        return tf.cast(h_flat, tf.complex64)

    def generate_channel(self, batch_size):
        if isinstance(batch_size, tf.Tensor):
            batch_size = tf.get_static_value(batch_size)
            if batch_size is None:
                raise ValueError("batch_size must be known at graph construction time")
        batch_size = int(batch_size)

        # Channel generation is pinned to CPU for robustness. In some TensorFlow
        # GPU setups, XLA/libdevice resolution can fail for control-flow kernels
        # used during topology sampling and TR 38.901 channel generation.
        with tf.device("/CPU:0"):
            # Sample scenario per sample
            scenario_idx = tf.random.uniform(
                [batch_size], minval=0, maxval=self.num_scenarios, dtype=tf.int32
            )

            # Output tensor (scatter blocks into it)
            h_out = tf.zeros(
                [batch_size, self.num_rx_antennas, self.num_tx_antennas],
                dtype=tf.complex64,
            )

            for si in range(self.num_scenarios):
                idxs = tf.where(tf.equal(scenario_idx, si))[:, 0]
                num_i = tf.shape(idxs)[0]

                def _true_fn(si=si, idxs=idxs):
                    return self._generate_scenario_block(si, idxs)

                def _false_fn():
                    return tf.zeros(
                        [0, self.num_rx_antennas, self.num_tx_antennas], tf.complex64
                    )

                h_block = tf.cond(num_i > 0, true_fn=_true_fn, false_fn=_false_fn)
                h_out = tf.tensor_scatter_nd_update(
                    h_out, tf.expand_dims(idxs, axis=1), h_block
                )

            return h_out

    def call(self, batch_size):
        return self.generate_channel(batch_size)


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
