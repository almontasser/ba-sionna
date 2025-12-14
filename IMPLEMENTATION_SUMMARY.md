# Implementation Summary: Sionna TR 38.901 Scenario Integration (UMi/UMa/RMa)

## Overview

This repo now uses **Sionna’s 3GPP TR 38.901 stochastic scenario channel models**
instead of CDL profiles. The beam-alignment pipeline and paper staging remain the
same (C3-only: UE RNN + BS FNN + learnable BS codebook), but the channel generator
is upgraded to sample realistic **UMi / UMa / RMa** conditions.

Key intent of the change:
- Remove the “fixed angles per CDL profile” artifact by sampling angles/delays from a
  random **topology** per sample.
- Preserve the paper’s **narrowband** measurement model by explicitly mapping the
  scenario CIR to a narrowband channel matrix \(H\).

## Paper-Critical Alignment Fixes

### 1) Narrowband mapping (CIR → narrowband \(H\))
The paper uses the narrowband scalar pilot measurement
\(y_t = w_t^H H f_t + w_t^H n_t\).

TR 38.901 scenario channels are frequency-selective (CIR). We therefore:
1. Sample CIR \((h, \tau)\) from Sionna (paths + delays)
2. Convert to CFR at one or more frequency offsets
3. Reduce CFR to a single narrowband \(H\)

Configured via:
- `Config.NARROWBAND_METHOD = "center"` (default, DC/center)
- `Config.NARROWBAND_METHOD = "subcarrier"` (pick `Config.NARROWBAND_SUBCARRIER`)
- `Config.NARROWBAND_METHOD = "mean_cfr"` (average over all subcarriers; for ablations)

Implementation: `channel_model.py` (`_cir_to_cfr`, `_reduce_to_narrowband`).

### 2) Satisfaction probability uses post-combining SNR (paper Eq. 4–6)
Satisfaction is computed from the receive SNR after combining:
\(\mathrm{SNR}_{RX} = \frac{|w^H H f|^2}{|w^H n|^2}\).

With unit-norm \(w\) and per-antenna SNR definition \(\mathrm{SNR}_{ANT}=1/\sigma_n^2\),
we use:
- `noise_power = sigma_n^2 = 1 / SNR_ANT`
- `SNR_RX(dB) = gain_dB - 10*log10(noise_power)`

Implementation: `metrics.py` and evaluation helpers in `figures_evaluators/common.py`.

## What Changed in the Code

### `channel_model.py`
- Removed CDL-based generator from the main pipeline.
- Implemented `SionnaScenarioChannelModel`:
  - Samples scenario **per sample** from `Config.SCENARIOS` (UMi/UMa/RMa).
  - Samples a simple 1-BS/1-UT topology per sample:
    - UT-BS distance range (`Config.DISTANCE_RANGE_M`)
    - UT velocity (`Config.UE_SPEED_RANGE`)
    - indoor/outdoor (`Config.INDOOR_PROBABILITY`)
    - scenario-dependent BS height (`Config.BS_HEIGHT_*`)
  - Calls Sionna’s system-level channel models (`UMi`, `UMa`, `RMa`)
  - Maps CIR → narrowband \(H\) for use in \(y_t\)

### `models/beam_alignment.py`
- Uses `SionnaScenarioChannelModel` to generate channels.
- Calls the channel model via `self.channel_model.generate_channel(batch_size)`.

### `config.py`
- Replaced CDL config with scenario config:
  - `SCENARIOS = ["UMi","UMa","RMa"]`
  - `O2I_MODEL`, `ENABLE_PATHLOSS`, `ENABLE_SHADOW_FADING`
  - topology ranges and heights
- `print_config()` reports scenario settings.

### `train.py`
- Uses `--scenarios "UMi,UMa,RMa"` instead of `--cdl_models`.
- Passes scenario config into `BeamAlignmentModel`.

### `evaluate.py` + `figures_evaluators/*`
- Figure-style plots now compare **scenarios** (UMi/UMa/RMa), keeping axes:
  - Beamforming gain vs SNR
  - Satisfaction probability vs SNR

Outputs:
- `results/figure_4_scenario_comparison.png`
- `results/figure_5_scenario_comparison_vs_T.png`

## How to Run

Train (C3-only):
```bash
python train.py -T 16 --epochs 100 --scenarios "UMi,UMa,RMa"
```

Evaluate:
```bash
python evaluate.py --figure 4 --num_samples 2000
```
