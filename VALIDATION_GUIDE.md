# Validation & Testing Guide (TR 38.901 Scenarios + Optional Mobility)

This repo implements the sensing/feedback loop from arXiv:2401.13587v1, but uses **Sionna’s 3GPP TR 38.901** stochastic scenario channels (**UMi/UMa/RMa**) and can optionally generate **time‑varying channels within an episode** (UE mobility).

If you are seeing `NaN`s, poor gain, or checkpoint restore errors, run the checks below first.

---

## 0) Quick sanity

```bash
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "import tensorflow as tf; print('GPUs', tf.config.list_physical_devices('GPU'))"
python -c "import sionna; print('Sionna', getattr(sionna,'__version__','?'))"
python -c "from config import Config; Config.print_config()"
```

---

## 1) Channel generation (static + time‑varying)

Static narrowband `H` (paper measurement model uses a single matrix):

```bash
python - <<'PY'
import tensorflow as tf
from channel_model import SionnaScenarioChannelModel

cm = SionnaScenarioChannelModel(
    num_tx_antennas=32,
    num_rx_antennas=16,
    carrier_frequency=28e9,
    scenarios=["UMi","UMa","RMa"],
    enable_pathloss=False,
    enable_shadow_fading=False,
    indoor_probability=0.0,
    narrowband_method="center",
)

H = cm.generate_channel(batch_size=128, num_time_samples=1)
print("H", H.shape, H.dtype)
print("finite:", bool(tf.reduce_all(tf.math.is_finite(tf.math.real(H))).numpy()))
print("E[||H||_F^2]:", float(tf.reduce_mean(tf.reduce_sum(tf.abs(H)**2, axis=(1,2))).numpy()))
PY
```

### Note on “CPU vs GPU” during training

Even if your NN forward/backward pass runs on GPU, **channel generation** can be a CPU bottleneck.
This repo exposes `Config.CHANNEL_GENERATION_DEVICE`:

- `"auto"` (default): try GPU if available
- `"gpu"`: force `/GPU:0` (falls back to CPU if no GPU is visible)
- `"cpu"`: force `/CPU:0`

Channel generation still uses Python control flow, so it can’t be “100% pure GPU”, but the heavy TF ops
inside Sionna can execute on GPU when you use `"auto"`/`"gpu"`.

For better GPU placement during training, this repo also defaults to:

- `Config.TRAIN_CHANNELS_OUTSIDE_GRAPH = True`

This generates `H` eagerly in the Python training loop and feeds it into the graph-compiled train step,
avoiding `tf.py_function` (which would otherwise force `H` to be produced on CPU inside `@tf.function`).

Time‑varying `H[t]` (only if you enable mobility in `config.py`):

```bash
python - <<'PY'
import tensorflow as tf
from channel_model import SionnaScenarioChannelModel

T = 16
cm = SionnaScenarioChannelModel(
    num_tx_antennas=32,
    num_rx_antennas=16,
    carrier_frequency=28e9,
    scenarios=["UMi"],
    enable_pathloss=False,
    enable_shadow_fading=False,
    indoor_probability=0.0,
    narrowband_method="center",
)

Hseq = cm.generate_channel(batch_size=8, num_time_samples=T+1, sampling_frequency=120e3)
print("Hseq", Hseq.shape, Hseq.dtype)  # (B, T+1, NRX, NTX)
print("finite:", bool(tf.reduce_all(tf.math.is_finite(tf.math.real(Hseq))).numpy()))
PY
```

---

## 2) End‑to‑end forward pass

```bash
python - <<'PY'
import tensorflow as tf
from config import Config
from models.beam_alignment import BeamAlignmentModel

model = BeamAlignmentModel(
    num_tx_antennas=Config.NTX,
    num_rx_antennas=Config.NRX,
    codebook_size=Config.NCB,
    num_sensing_steps=Config.T,
    rnn_hidden_size=Config.RNN_HIDDEN_SIZE,
    rnn_type=Config.RNN_TYPE,
    num_feedback=Config.NUM_FEEDBACK,
    carrier_frequency=Config.CARRIER_FREQUENCY,
    scenarios=Config.SCENARIOS,
)

res = model(batch_size=16, snr_db=10.0, training=False)
print("channels:", res['channels'].shape)
print("final_tx_beams:", res['final_tx_beams'].shape)
print("final_rx_beams:", res['final_rx_beams'].shape)
print("beamforming_gain:", res['beamforming_gain'].shape)
print("received_signals:", res['received_signals'].shape)
print("beam_indices:", res['beam_indices'].shape)
if getattr(Config, 'MOBILITY_ENABLE', False):
    print("channels_sequence:", res['channels_sequence'].shape)
PY
```

---

## 3) Training step smoke test (finite loss / no NaNs)

```bash
python - <<'PY'
import tensorflow as tf
from config import Config
from train import create_model, train_step

model = create_model(Config)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

# build
_ = model(batch_size=32, snr_db=10.0, training=True)

for i in range(10):
    H, _, _ = model.generate_channels(32)
    loss, bf_gain_db, grad_norm, update_skipped = train_step(
        model,
        opt,
        batch_size=32,
        snr_db=tf.constant(10.0, tf.float32),
        channels=H,
    )
    gain_norm = -loss  # for LOSS_TYPE='paper'
    print(
        i,
        "loss",
        float(loss.numpy()),
        "gain_norm",
        float(gain_norm.numpy()),
        "BF_gain_dB",
        float(bf_gain_db.numpy()),
        "grad_norm",
        float(grad_norm.numpy()),
        "update_skipped",
        int(update_skipped.numpy()),
    )
PY
```

### Interpreting the logs (common confusion)

With `LOSS_TYPE="paper"`, the loss is the **negative mean normalized gain**:

\[
\text{gain\_norm} = \\frac{|w^H H f|^2}{\\lVert H\\rVert_F^2} \\,\\in [0,1]
\\quad\\Rightarrow\\quad
\\text{loss} = -\\mathbb{E}[\\text{gain\_norm}]
\]

So:
- **More negative loss is better.** (`-0.20` is better than `-0.01`.)
- The training bar prints `gain_norm ≈ -loss` to avoid sign confusion.
- `BF_gain` is **absolute** \(10\\log_{10}(|w^H H f|^2)\). It can be negative if the effective gain is `< 1` (especially if you enable pathloss / long distances).

---

## 4) Scheme compliance tests

```bash
python test_scheme_compliance.py
```

This checks:
- output shapes,
- gradient flow through all trainables,
- codebook variables update under optimization,
- and (if enabled) that mobility produces `channels_sequence` with `T+1` samples.

---

## 5) Checkpoint restore issues (shape mismatch)

If you see errors like:
- `Received incompatible tensor with shape ... when attempting to restore variable with shape ...`

That means the checkpoint was created from a **different model definition** (e.g., different UE feature encoding, RNN size, added LayerNorm, mobility changes).

Current behavior:
- `train.py` checks checkpoint compatibility and will automatically switch to a fresh directory if it finds an incompatible checkpoint.
- `evaluate.py` will raise a clear error if the checkpoint does not match the current model.

Recommended fix:
- Train from scratch in a new directory: `python train.py --checkpoint_dir ./checkpoints_new_run`
