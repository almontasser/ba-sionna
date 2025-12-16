# TensorBoard (How to Use)

This repo writes TensorBoard event files during training so you can monitor loss, normalized gain, beamforming gain, gradients, and validation metrics.

## 1) Install

`tensorboard` is already listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 2) Train (creates logs)

```bash
python train.py --checkpoint_dir ./checkpoints_run1 --log_dir ./logs
```

Training creates a per-run directory like:

```
./logs/run_YYYYMMDD-HHMMSS/
```

## 3) Start TensorBoard

```bash
tensorboard --logdir ./logs --port 6006
```

Open:

```
http://localhost:6006
```

### If training runs on a remote machine

Start TensorBoard on the remote host:

```bash
tensorboard --logdir ./logs --port 6006 --bind_all
```

Then forward the port from your laptop:

```bash
ssh -L 6006:localhost:6006 user@remote_host
```

Open `http://localhost:6006` locally.

## 4) What you will see

Scalars are logged under these tags:

- `train/loss` (negative normalized gain; more negative is better)
- `train/gain_norm` (≈ `-train/loss`, in [0,1])
- `train/bf_gain_db` (absolute beamforming gain in dB)
- `train/gradient_norm`
- `train/learning_rate`
- `val/loss`, `val/bf_gain_db`, `val/satisfaction_prob`

The full printed config is saved once per run under:

- Text: `run/config`

