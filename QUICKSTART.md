# Quickstart (TR 38.901 Scenarios + Optional Mobility)

This repo implements the sensing/feedback loop from arXiv:2401.13587v1 using:
- **Sionna 3GPP TR 38.901** stochastic scenarios (UMi/UMa/RMa)
- optional **time‑varying channels within an episode** (UE mobility)

## 1) Install + sanity

```bash
pip install -r requirements.txt
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "import sionna; print('Sionna', getattr(sionna,'__version__','?'))"
python config.py
```

## 2) Run smoke tests (recommended before training)

```bash
python test_scheme_compliance.py
```

## 3) Train (small test run)

```bash
python train.py --test_mode --checkpoint_dir ./checkpoints_smoke
```

If you previously trained with a different architecture and hit a restore error,
use a fresh directory (or delete old checkpoints). See `VALIDATION_GUIDE.md`.

## 4) Train (normal run)

```bash
python train.py --checkpoint_dir ./checkpoints_run1
```

## 5) Evaluate (paper-style plots)

```bash
python evaluate.py --figure all --num_samples 2000
```

## Useful docs

- `VALIDATION_GUIDE.md` (NaNs, loss vs gain, checkpoint issues)
- `IMPROVEMENTS_OVER_ARXIV_2401_13587v1.md` (what differs vs the paper baseline)
- `MOBILITY_TR38901_TIME_VARIATION.md` (how mobility is modeled here)
- `NN_ARCHITECTURE_AND_TUNING.md` (RNN/FNN sizing and tuning notes)

