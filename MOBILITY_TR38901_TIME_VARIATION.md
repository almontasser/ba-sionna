# Mobility / Time-Varying TR 38.901 Channels

This repo originally sampled **one** TR 38.901 channel snapshot per episode and used it for all sensing steps.
That makes `UE_SPEED_RANGE` mostly irrelevant (because Doppler/time evolution needs multiple time samples).

This document describes the implemented upgrade: **time-varying channels within an episode** to model UE mobility.

## What “mobility” means here

- The UE has a **random velocity vector** (m/s) sampled per channel realization.
- The TR 38.901 system-level channel generator is asked for **multiple time samples** (`num_time_samples > 1`).
- The sensing loop uses **H[t] at step t**, and the final beamforming gain is evaluated on **H[T]**.

This is consistent with using the TR 38.901 stochastic model with Doppler/time evolution from UE velocity.

## Where it is implemented

- Channel generation (TR 38.901 UMi/UMa/RMa):
  - `channel_model.py` (`SionnaScenarioChannelModel`)
  - `set_topology(... ut_velocities=...)` is populated from the sampled UE speed/range.
  - `sl(num_time_samples=..., sampling_frequency=...)` generates time evolution.
- Beam-alignment loop uses time-varying H:
  - `models/beam_alignment.py`:
    - sensing: `y_t = w_t^H H[t] f_t + noise`
    - final objective: gain computed with `H[T]`

## Configuration knobs

In `config.py`:

- `Config.MOBILITY_ENABLE` (bool)
  - If `True`, the model requests `T+1` time samples per episode and uses time-varying H.
- `Config.MOBILITY_NUM_TIME_SAMPLES` (int or `None`)
  - If `None`, uses `T+1`.
- `Config.MOBILITY_SAMPLING_FREQUENCY_HZ` (float)
  - Defines the time step `dt = 1 / f_s` used by the TR 38.901 time evolution.
  - Default uses `RESOURCE_GRID_SUBCARRIER_SPACING`, i.e., `dt ≈ 1/subcarrier_spacing` (OFDM-symbol scale).

## Limitations / interpretation

- This models **time evolution via Doppler** over the sampled time grid within one episode.
- It does not explicitly simulate long trajectories with path birth/death or re-dropping large-scale parameters.
  If you want long-horizon motion, you typically need a spatial-consistency procedure and/or RT-based motion.

## References (papers + docs)

The following references guided this implementation:

- Paper under implementation: https://arxiv.org/html/2401.13587v1
- 3GPP TR 38.901 official spec archive (all versions, zip): https://www.3gpp.org/ftp/specs/archive/38_series/38.901
- Sionna TR 38.901 UMi model implementation (time samples + sampling frequency):
  https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/umi.html
- Sionna wireless channel model API (UMi/UMa/RMa inputs, incl. `ut_velocities` and time sampling):
  https://nvlabs.github.io/sionna/phy/api/channel.wireless.html
- Sionna tutorial (time-varying channels via `num_time_samples` and `sampling_frequency`):
  https://nvlabs.github.io/sionna/phy/tutorials/Sionna_tutorial_part3.html
- Jaeckel et al., “Efficient Sum-of-Sinusoids based Spatial Consistency for the 3GPP New-Radio Channel Model”:
  https://arxiv.org/abs/1808.04659
- Hoydis et al., “Sionna: An Open-Source Library for Next-Generation Physical Layer Research”:
  https://arxiv.org/abs/2203.11854
- NIST: analysing the TR 38.901 spatial consistency procedure:
  https://www.nist.gov/publications/anaylsing-3gpp-spatial-consistency-procedure-through-channel-measurements
