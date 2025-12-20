# Hyperparameter References (Beam Alignment / mmWave)

This file lists learning-rate and batch-size settings reported in related
mmWave beamforming / beam-alignment literature. These are example values,
not "perfect" settings, and should be re-tuned for your dataset, model size,
and SNR range.

## Reported settings

| Source | Task/Model | Learning rate | Batch size | Other notes |
| --- | --- | --- | --- | --- |
| MDPI Sensors 2023, DRL-based coordinated beamforming for mmWave massive MIMO vehicular networks | DRL beamforming | Adam LR = 0.0005 | 96 | 250 episodes |
| MDPI Electronics 2025, BeamSecure-AI (beam-level attack detection in mmWave RAN) | RL/AI detection | Adam LR = 1e-4 | 64 | 60 epochs; early stopping |
| MDPI Electronics 2022, Hybrid Beamforming for MISO via CNN | CNN beamforming | LR = 0.01 (decayed) | Not specified | 1000 epochs; 16 batches/epoch |

## Notes

- These papers use different tasks (beamforming, detection, or related RL),
  so treat them as anchors for a reasonable range, not targets.
- Start with LR in the 1e-4 to 1e-3 range for Adam and adjust with LR sweeps.

## Best Overall Recommendation (for this repo)

**Use an LR range test first, then train with a periodic schedule.**

Why this is best overall:
- The LR range test finds a stable LR band for *this* model/data.
- A periodic schedule (e.g., cosine restarts) helps escape plateaus without
  guessing a single fixed LR.

Practical steps:
1) Run a short LR range test (increasing LR over a few thousand steps).
2) Pick a stable LR band from the loss curve (avoid divergence).
3) Use cosine restarts with `initial_lr` near the top of the stable band.

Note: The LR range test is implemented in `train.py` via `--lr_range_test`.

## References

1) https://www.mdpi.com/1424-8220/23/5/2772  
2) https://www.mdpi.com/2079-9292/14/23/4642  
3) https://www.mdpi.com/2079-9292/11/14/2213
