# Improvements Over arXiv:2401.13587v1 (Paper Implementation Assumptions)

This document explains how *this repository’s implementation* extends and improves on the baseline assumptions commonly used when implementing **“Deep Learning Based Adaptive Joint mmWave Beam Alignment” (arXiv:2401.13587v1)**.

Paper (HTML): https://arxiv.org/html/2401.13587v1

The goal is not to change the core idea of the paper (adaptive sensing + feedback, trained end-to-end), but to make the system **more realistic**, **more robust**, and **easier to validate/reproduce** in a standards-based setting.

---

## What We Keep Faithful to the Paper

This repo keeps the paper’s key *algorithmic structure*:

- **Sensing loop** of length `T`: BS transmits a probing beam `f_t`, UE applies a combining vector `w_t`, and measures a scalar pilot
  \[
  y_t = w_t^H H f_t + w_t^H n_t
  \]
- **UE controller** is an RNN that updates its state from the history of measurements and BS beam indices, and outputs:
  - next combining vector `w_t`
  - a feedback message `m_t`
- **BS controller** performs codebook-based sweeping during sensing and uses an FNN to map final feedback to the final transmit beam.

Where to see this in code:
- End-to-end flow: `models/beam_alignment.py`
- UE RNN input features: `models/ue_controller.py`
- BS sweeping + feedback FNN + learnable codebook: `models/bs_controller.py`

---

## Key Improvements in This Repository

### 1) Standards-based channels (TR 38.901) instead of a simplified geometric channel

Many “paper-faithful” implementations use a simple L-path geometric narrowband channel model with randomly drawn AoA/AoD and Rayleigh path gains. That model is fast and clean, but it can hide real-world effects (topology dependence, scenario statistics, LOS/NLOS structure, mobility effects).

This repo upgrades the channel generator to **Sionna’s 3GPP TR 38.901 stochastic scenario models**:

- Scenario family support: **UMi / UMa / RMa** (configurable)
- Per-sample topology sampling (distance, heights, indoor/outdoor state, UE velocity)
- Realistic path-angle and delay statistics implied by the standard

Where:
- Scenario channel generator: `channel_model.py` (`SionnaScenarioChannelModel`)
- Scenario selection & topology sampling per sample: `channel_model.py` (`generate_channel()`, `_sample_topology()`)

Why it’s an improvement:
- Trains and evaluates on **more realistic angular statistics** (AoA/AoD are not drawn from a simplistic independent uniform model).
- Supports **scenario generalization** (e.g., training on UMi/UMa/RMa together).
- Makes results more meaningful for “deployment-like” conditions.

### 2) Explicit CIR → narrowband \(H\) mapping (paper measurement model preserved)

TR 38.901 channels are inherently **frequency-selective** (CIR with delays). The paper’s sensing measurement uses a **narrowband matrix \(H\)**.

This repo preserves the paper’s measurement model by explicitly mapping:

1. Sionna CIR \((h,\tau)\) → CFR at selected frequency offset(s)
2. CFR → a **single narrowband** \(H\) used by the sensing equation

Where:
- Mapping utilities: `channel_model.py` (`_cir_to_cfr()`, `_reduce_to_narrowband()`)
- Configuration: `config.py` (`NARROWBAND_METHOD`, `NARROWBAND_SUBCARRIER`)

Why it’s an improvement:
- Avoids silently “flattening” a frequency-selective channel in a way that is hard to reason about.
- Makes it clear what “narrowband” means when using a standards channel.

### 3) Domain randomization across scenario and SNR (robustness-oriented training)

The paper often reports results under a fixed/controlled training setup (e.g., one environment, one SNR setting). In practice, beam alignment must work across a range of operating points.

This repo supports training with **domain randomization**, including:
- Random scenario selection per sample (from `Config.SCENARIOS`)
- Random SNR per batch (from `Config.SNR_TRAIN_RANGE` when enabled)

Where:
- SNR sampling: `train.py` (`sample_snr()`)
- SNR settings: `config.py` (`SNR_TRAIN_RANDOMIZE`, `SNR_TRAIN_RANGE`)

Why it’s an improvement:
- Reduces “overfitting to one clean SNR” and improves robustness to noise changes.
- Makes it easy to reproduce both “fixed-SNR” training and “robust” training by flipping config flags.

### 4) Better verification: measurement ablations + figure-style evaluations

In adaptive sensing, it is easy to accidentally build a system that “works” but does not truly use the sensing measurements `y_t` (e.g., it learns a near-static sweep policy).

This repo includes tooling to verify behavior:

- **Measurement ablation runner**:
  - `none`: normal measurements
  - `zero`: force `y_t = 0`
  - `noise_only`: keep only noise
  - `shuffle`: break the association between `H` and `y_t`
  - File: `run_measurement_ablation.py`
- **Figure-style evaluation scripts** to generate scenario-vs-SNR and scenario-vs-T curves:
  - Entry point: `evaluate.py`
  - Helpers: `figures_evaluators/`

Why it’s an improvement:
- Gives you a concrete way to answer: “Is my trained agent actually using the measurement channel?”
- Makes comparisons repeatable and consistent with paper-style plots.

### 5) Engineering robustness: run reliably under modern TF + Sionna

Some TF/Sionna combinations can fail when TR 38.901 channel code is traced under `@tf.function` (graph mode). This repo routes TR 38.901 channel generation through an eager-only path when needed.

Where:
- Channel graph-compatibility routing: `channel_model.py` (`generate_channel()` via `tf.py_function`)
- Graph-safe BS codebook initialization (no `.numpy()` in `build()`): `models/bs_controller.py`

Why it’s an improvement:
- Training runs are less likely to break due to graph tracing limitations inside third-party channel code.

---

## Important Notes / Trade-offs

- Using `tf.py_function` for channel generation can reduce performance vs fully-graph channel generation (but it avoids hard failures and keeps the rest of training graph-compiled).
- The repo defaults to **paper-style normalization** (e.g., pathloss disabled by default). If you enable pathloss, SNR interpretation and distributions change; treat it as a different experiment.

---

## If You Want the “Most Paper-Like” Setup Here

Use the repo’s current structure but reduce domain randomization:

- Fix SNR training: set `Config.SNR_TRAIN_RANDOMIZE = False` and `Config.SNR_TRAIN = <value>`
- Use a single scenario: set `Config.SCENARIOS = ["UMi"]` (or whichever you want)
- Keep narrowband mapping at the center subcarrier: `Config.NARROWBAND_METHOD = "center"`

This yields a controlled setting closer to the typical paper baseline, while keeping the implementation modular and standards-capable.
