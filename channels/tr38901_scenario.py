"""
Sionna-based 3GPP TR 38.901 stochastic scenario channels (UMi/UMa/RMa).

This module provides a wrapper that samples a topology (BS/UT locations, heights,
indoor/outdoor state, UE velocity) and calls Sionna's TR 38.901 system-level
models to generate a channel impulse response. The CIR is then explicitly mapped
to a narrowband H used by the paper's sensing equation.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .narrowband import cir_to_cfr, frequency_offsets_hz, reduce_to_narrowband

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


class SionnaScenarioChannelModel(tf.keras.layers.Layer):
    """
    TR 38.901 stochastic scenario channel (UMi/UMa/RMa), mapped to narrowband H.

    Public interface:
      - generate_channel(batch_size, num_time_samples=1, sampling_frequency=1.0)
          -> (B, NRX, NTX) if num_time_samples==1
          -> (B, S, NRX, NTX) if num_time_samples==S>1
    """

    def __init__(
        self,
        num_tx_antennas: int,
        num_rx_antennas: int,
        carrier_frequency: float = 28e9,
        scenarios: list[str] | None = None,
        o2i_model: str = "low",  # required for UMi/UMa ("low" or "high")
        enable_pathloss: bool = False,
        enable_shadow_fading: bool = False,
        distance_range_m: tuple[float, float] = (10.0, 200.0),
        ue_speed_range: tuple[float, float] = (0.0, 30.0),
        indoor_probability: float = 0.0,
        ut_height_m: float = 1.5,
        bs_height_umi_m: float = 10.0,
        bs_height_uma_m: float = 25.0,
        bs_height_rma_m: float = 35.0,
        fft_size: int = 64,
        subcarrier_spacing: float = 120e3,
        narrowband_method: str = "center",
        narrowband_subcarrier: int | None = None,
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
                raise ValueError(f"Unknown scenario '{s}'. Expected one of: UMi, UMa, RMa.")

        self._f_offsets = frequency_offsets_hz(
            self.narrowband_method,
            self.fft_size,
            self.subcarrier_spacing,
            self.narrowband_subcarrier,
        )

        print("✓ Sionna Scenario Channel Model initialized (TR 38.901)")
        print(f"  - Scenarios: {', '.join(self.scenarios)}")
        print(f"  - Carrier frequency: {self.carrier_frequency/1e9:.1f} GHz")
        print(f"  - Distance range: {self.distance_range_m[0]:.1f}-{self.distance_range_m[1]:.1f} m")
        print(f"  - UE speed range: {self.ue_speed_range[0]:.1f}-{self.ue_speed_range[1]:.1f} m/s")
        print(f"  - Indoor probability: {self.indoor_probability:.2f} (used for UMi/UMa O2I)")
        print(f"  - OFDM grid: fft_size={self.fft_size}, subcarrier_spacing={self.subcarrier_spacing/1e3:.0f} kHz")
        nb_desc = self.narrowband_method
        if self.narrowband_method == "subcarrier":
            nb_desc += f" (k={self.narrowband_subcarrier})"
        print(f"  - Narrowband method: {nb_desc}")
        print(f"  - Antennas: {self.num_tx_antennas} BS, {self.num_rx_antennas} UE")

    def _bs_height_for_scenario(self, scenario: str) -> float:
        if scenario == "UMi":
            return self.bs_height_umi_m
        if scenario == "UMa":
            return self.bs_height_uma_m
        return self.bs_height_rma_m

    def _sample_topology(self, num_samples: tf.Tensor, scenario: str):
        """Sample a (num_samples,1,*) topology for one UT and one BS."""
        num_samples = tf.cast(num_samples, tf.int32)
        d_min, d_max = self.distance_range_m
        d_min = tf.constant(d_min, tf.float32)
        d_max = tf.constant(d_max, tf.float32)

        # Uniform-in-area radius sampling in [d_min, d_max]
        u = tf.random.uniform([num_samples], 0.0, 1.0, dtype=tf.float32)
        d_max_sq = d_max * d_max
        d_min_sq = d_min * d_min
        r = tf.sqrt(u * (d_max_sq - d_min_sq) + d_min_sq)
        phi = tf.random.uniform([num_samples], 0.0, 2.0 * 3.141592653589793, tf.float32)

        x = r * tf.cos(phi)
        y = r * tf.sin(phi)

        bs_h = tf.constant(self._bs_height_for_scenario(scenario), tf.float32)
        ut_h = tf.constant(self.ut_height_m, tf.float32)

        bs_loc = tf.stack(
            [tf.zeros_like(x), tf.zeros_like(y), tf.fill([num_samples], bs_h)], axis=-1
        )
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

    def _generate_scenario_block(
        self, scenario_idx: int, num_samples: int, num_time_samples: int, sampling_frequency: float
    ) -> tf.Tensor:
        scenario = self.scenarios[scenario_idx]
        num_i = tf.cast(num_samples, tf.int32)
        num_time_samples = int(num_time_samples)
        sampling_frequency = float(sampling_frequency)
        if num_time_samples <= 0:
            raise ValueError("num_time_samples must be a positive integer.")

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

        # Time evolution / mobility:
        # Sionna's TR 38.901 system-level models support multiple time samples via
        # Doppler induced by `ut_velocities` and the provided sampling frequency.
        #
        # References (URLs added per request):
        # - Sionna TR38901 UMi implementation docs:
        #   https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/umi.html
        # - Sionna tutorial (time sampling / time-varying channels):
        #   https://nvlabs.github.io/sionna/phy/tutorials/Sionna_tutorial_part3.html
        # - 3GPP TR 38.901 official spec archive (all versions, zip):
        #   https://www.3gpp.org/ftp/specs/archive/38_series/38.901
        # - Jaeckel et al., efficient spatial consistency for 3GPP NR channels:
        #   https://arxiv.org/abs/1808.04659
        # - NIST: analysing the TR 38.901 spatial consistency procedure:
        #   https://www.nist.gov/publications/anaylsing-3gpp-spatial-consistency-procedure-through-channel-measurements
        h_cir, tau = sl(num_time_samples=num_time_samples, sampling_frequency=sampling_frequency)

        # Squeeze singleton dims:
        #   h_cir: (B, 1, NRX, 1, NTX, P, S) -> (B, NRX, NTX, P, S)
        #   tau  : (B, 1, 1, P)             -> (B, P)
        h_cir_s = tf.squeeze(h_cir, axis=[1, 3])
        tau_s = tf.squeeze(tau, axis=[1, 2])

        # CIR -> CFR -> narrowband H
        h_freq = cir_to_cfr(h_cir_s, tau_s, self._f_offsets)
        h_flat = reduce_to_narrowband(h_freq, self.narrowband_method)

        if num_time_samples == 1:
            # (B, NRX, NTX, 1) -> (B, NRX, NTX) for backwards compatibility
            h_flat = tf.squeeze(h_flat, axis=-1)
            return tf.cast(h_flat, tf.complex64)

        # Return as (B, S, NRX, NTX) to match time-major access in the sensing loop.
        h_seq = tf.transpose(h_flat, perm=[0, 3, 1, 2])
        return tf.cast(h_seq, tf.complex64)

    def _generate_channel_eager(self, batch_size, num_time_samples=1, sampling_frequency=1.0):
        """
        Eager-only channel generation.

        Why this exists:
          Some Sionna TR 38.901 scenario implementations use Python control flow
          that is incompatible with TensorFlow graph tracing in certain TF/Sionna
          combinations (e.g., Python `if` on a symbolic Tensor). To keep training
          working under `@tf.function`, `generate_channel()` routes graph-mode
          calls through `tf.py_function`, which executes this method eagerly.
        """
        batch_size = int(batch_size)
        num_time_samples = int(num_time_samples)
        sampling_frequency = float(sampling_frequency)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if num_time_samples <= 0:
            raise ValueError("num_time_samples must be a positive integer.")

        with tf.device("/CPU:0"):
            scenario_idx = np.random.randint(
                0, self.num_scenarios, size=batch_size, dtype=np.int32
            )
            if num_time_samples == 1:
                h_out = tf.zeros(
                    [batch_size, self.num_rx_antennas, self.num_tx_antennas],
                    dtype=tf.complex64,
                )
            else:
                h_out = tf.zeros(
                    [batch_size, num_time_samples, self.num_rx_antennas, self.num_tx_antennas],
                    dtype=tf.complex64,
                )

            for si in range(self.num_scenarios):
                idxs = np.nonzero(scenario_idx == si)[0]
                if idxs.size == 0:
                    continue

                # Generate and sanitize a block. Some TF/Sionna combinations can
                # occasionally emit non-finite values; resample a few times and
                # finally zero any remaining non-finite entries.
                h_block = None
                for _ in range(3):
                    candidate = self._generate_scenario_block(
                        int(si),
                        int(idxs.size),
                        num_time_samples,
                        sampling_frequency,
                    )
                    finite_mask = tf.math.is_finite(tf.math.real(candidate)) & tf.math.is_finite(
                        tf.math.imag(candidate)
                    )
                    if bool(tf.reduce_all(finite_mask).numpy()):
                        h_block = candidate
                        break
                if h_block is None:
                    h_block = candidate
                finite_mask = tf.math.is_finite(tf.math.real(h_block)) & tf.math.is_finite(
                    tf.math.imag(h_block)
                )
                h_block = tf.where(finite_mask, h_block, tf.zeros_like(h_block))
                scatter_idx = tf.constant(idxs.reshape(-1, 1), dtype=tf.int32)
                h_out = tf.tensor_scatter_nd_update(h_out, scatter_idx, h_block)

            finite_mask = tf.math.is_finite(tf.math.real(h_out)) & tf.math.is_finite(tf.math.imag(h_out))
            h_out = tf.where(finite_mask, h_out, tf.zeros_like(h_out))
            return h_out

    def generate_channel(self, batch_size, num_time_samples=1, sampling_frequency=1.0):
        if tf.executing_eagerly():
            return self._generate_channel_eager(batch_size, num_time_samples, sampling_frequency)

        bs_tensor = (
            batch_size
            if isinstance(batch_size, tf.Tensor)
            else tf.constant(int(batch_size), tf.int32)
        )
        nt_tensor = (
            num_time_samples
            if isinstance(num_time_samples, tf.Tensor)
            else tf.constant(int(num_time_samples), tf.int32)
        )
        fs_tensor = (
            sampling_frequency
            if isinstance(sampling_frequency, tf.Tensor)
            else tf.constant(float(sampling_frequency), tf.float32)
        )

        h = tf.py_function(
            func=lambda bs, nt, fs: self._generate_channel_eager(
                int(bs.numpy()),
                int(nt.numpy()),
                float(fs.numpy()),
            ),
            inp=[bs_tensor, nt_tensor, fs_tensor],
            Tout=tf.complex64,
        )

        static_bs = tf.get_static_value(bs_tensor)
        static_nt = tf.get_static_value(nt_tensor)
        if static_bs is None:
            if static_nt is None:
                h.set_shape([None, None, self.num_rx_antennas, self.num_tx_antennas])
            elif int(static_nt) == 1:
                h.set_shape([None, self.num_rx_antennas, self.num_tx_antennas])
            else:
                h.set_shape([None, int(static_nt), self.num_rx_antennas, self.num_tx_antennas])
        else:
            if static_nt is None:
                h.set_shape([int(static_bs), None, self.num_rx_antennas, self.num_tx_antennas])
            elif int(static_nt) == 1:
                h.set_shape([int(static_bs), self.num_rx_antennas, self.num_tx_antennas])
            else:
                h.set_shape([int(static_bs), int(static_nt), self.num_rx_antennas, self.num_tx_antennas])
        return h

    def call(self, batch_size):
        return self.generate_channel(batch_size)

