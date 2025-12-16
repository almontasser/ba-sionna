# Neural Network / RNN Architecture (C3) and Tuning Notes

This document describes the NN/RNN architecture used in this repo’s **C3-only** implementation (paper: https://arxiv.org/html/2401.13587v1), and what we changed to make it more suitable for **multi-scenario TR 38.901** channels and **within-episode mobility**.

## Where the “networks” are in code

- End-to-end loop: `models/beam_alignment.py`
- UE controller (N1): `models/ue_controller.py`
- BS controller (N2 + N3): `models/bs_controller.py`
- Config knobs: `config.py`

## Current defaults (as of this repo state)

All defaults below come from `config.py`.

### UE controller (N1)

**Purpose:** Given the sensing history, generate the receive combining vector and the feedback message.

Implementation summary (`models/ue_controller.py`):

- RNN type: `Config.RNN_TYPE` (default `"GRU"`)
- RNN layers: `Config.RNN_NUM_LAYERS` (default `2`)
- Hidden size per layer: `Config.RNN_HIDDEN_SIZE` (default `256`)
- Input features per step:
  - `Re(y_t)` and `Im(y_t)`
  - beam index feature for `x_t`:
    - `Config.UE_BEAM_INDEX_ENCODING = "one_hot"` (default) or `"scalar"`
  - optional SNR feature:
    - `Config.UE_INCLUDE_SNR_FEATURE = True` (default): adds `snr_db * Config.UE_SNR_FEATURE_SCALE`
  - optional time feature:
    - `Config.UE_INCLUDE_TIME_FEATURE = True` (default): adds `t/(T-1)`
  - optional input layer norm:
    - `Config.UE_INPUT_LAYER_NORM = True` (default)
- Output heads (from last-layer hidden state):
  - Beam head: Dense → `2*NRX` real outputs → complex vector → unit-norm (`normalize_beam`)
  - Feedback head: Dense → `NFB` real outputs

Why these changes were made for the “current problem” (TR 38.901 + mobility):

- **Beam index as one-hot** often learns more reliably than a single scalar index when the policy must generalize across scenarios.
- **Time-step feature** helps the RNN disambiguate early vs late sensing decisions when the channel evolves within an episode.
- **LayerNorm on inputs** reduces sensitivity to scale shifts in `y_t` across scenarios/SNRs.

### BS controller (N2 + N3)

**N3 (learned codebook):**

- Trainable complex codebook of size `NCB × NTX`, initialized from a DFT codebook.
- Used for the sensing sweep: indices `(start+t) mod NCB`.

**N2 (feedback network):**

- FNN mapping feedback `m_FB` → final transmit beam `f_T`.
- Configurable via:
  - `Config.BS_FNN_HIDDEN_SIZES` (default `(256, 256)`)
  - `Config.BS_FNN_ACTIVATION` (default `"gelu"`)
  - `Config.BS_FNN_LAYER_NORM` (default `True`)
- Output is normalized to unit-norm before use.

## Mobility-specific note (why RNN design matters more now)

With `Config.MOBILITY_ENABLE=True`, the channel model produces a **time-varying** sequence `H[t]` and the sensing loop uses `H[t]` at step `t` (see `models/beam_alignment.py`).

This makes the policy partially a **tracking** problem:
- the RNN must integrate noisy measurements while the underlying channel changes,
- so capacity and input feature design matter more than in the quasi-static setting.

See `MOBILITY_TR38901_TIME_VARIATION.md` for details.

## Parameter counts (sanity check)

With the current defaults (`GRU`, 2 layers, hidden size 256, one-hot `x_t`, input LayerNorm), the end-to-end model is roughly **0.7M trainable parameters**.

If you increase `RNN_HIDDEN_SIZE` substantially (e.g., 384), the UE RNN alone can exceed **1M parameters**, which is usually not a good default when training stability and iteration speed matter.

## Recommended tuning profiles

### 1) “Paper-like / easier training” profile (quasi-static)

If you want something closer to the original problem difficulty (channel constant within an episode):

- `Config.MOBILITY_ENABLE = False`
- Keep the RNN small-ish:
  - `Config.RNN_NUM_LAYERS = 2`
  - `Config.RNN_HIDDEN_SIZE = 256` (or 192/128 if you need speed)
- You can keep `UE_BEAM_INDEX_ENCODING="one_hot"`; it doesn’t break the scheme.
- `UE_INCLUDE_TIME_FEATURE` can be `False` (optional).

### 2) “Mobility / time-varying” profile

For time-varying channels:

- `Config.MOBILITY_ENABLE = True`
- Keep:
  - `UE_INCLUDE_TIME_FEATURE = True`
  - `UE_BEAM_INDEX_ENCODING = "one_hot"`
  - `UE_INPUT_LAYER_NORM = True`
- If learning is unstable:
  - try a smaller SNR range during early training (curriculum),
  - reduce `UE_SPEED_RANGE` or reduce the per-step time increment by increasing `MOBILITY_SAMPLING_FREQUENCY_HZ`.
