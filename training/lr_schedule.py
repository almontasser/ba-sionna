"""
Learning-rate schedules used for training.
"""

from __future__ import annotations

import tensorflow as tf


class WarmupThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up to `base_lr`, then exponential decay.
    """

    def __init__(self, base_lr: float, warmup_steps: int, decay_steps: int, decay_rate: float):
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.decay = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=float(base_lr),
            decay_steps=int(decay_steps),
            decay_rate=float(decay_rate),
            staircase=True,
        )

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        if self.warmup_steps > 0:
            warm_lr = self.base_lr * tf.cast(step, tf.float32) / tf.cast(
                self.warmup_steps, tf.float32
            )
            return tf.where(step < self.warmup_steps, warm_lr, self.decay(step - self.warmup_steps))
        return self.decay(step)

