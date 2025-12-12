# GPU Performance Analysis & Optimization Guide

## Current Status

✅ **GPU Detection**: Working (Tesla T4 detected)  
⚠️ **Training Speed**: Slow (~1.13 it/s with batch size 256)  
ℹ️ **Note**: Repo is C3-only (no C1/C2 scheme branches)

## Performance Bottlenecks Identified

### 1. Sequential Loop Processing (Major Impact)
**Location**: `models/beam_alignment.py`, lines 201-234

The beam alignment executes **16 sequential sensing steps** in a Python for-loop:
```python
for t in range(T):  # T=16, runs sequentially
    # Get beams, compute signals, update RNN state
```

**Impact**: Each iteration waits for the previous to complete, preventing GPU parallelization.

**Estimated Performance Impact**: 🔴 **Very High** (likely the main bottleneck)

### 2. Complex Number Operations
**Location**: Throughout the codebase

Operations like `tf.linalg.matvec`, `tf.math.conj`, complex multiplication may not be fully optimized on GPU.

**Impact**: Moderate, especially with large batch sizes.

**Estimated Performance Impact**: 🟡 **Medium**

### 3. RNN Sequential Dependencies
**Location**: `models/ue_controller.py`

RNN states must be computed sequentially (inherent to RNN architecture).

**Impact**: Cannot be parallelized due to sequential nature of RNNs.

**Estimated Performance Impact**: 🟡 **Medium** (unavoidable for RNN)

### 4. Channel Generation
**Location**: `channel_model.py`

Geometric channel generation involves random sampling and array operations.

**Impact**: Low - happens once per batch.

**Estimated Performance Impact**: 🟢 **Low**

---

## Optimization Recommendations

### 🚀 High Priority (Significant Speedup Expected)

#### 1. Vectorize the Sensing Loop
**Current**: Sequential for-loop  
**Proposed**: Batched matrix operations

**Complexity**: High (requires architectural changes)  
**Expected Speedup**: 3-5x

The challenging part is that RNN state updates are inherently sequential. However, we can:
- Pre-compute all TX beams (already done)
- Use TensorFlow's built-in RNN batching
- Vectorize the signal computation

See implementation suggestion in next section.

#### 2. Use Mixed Precision Training
**Implementation**: Enable automatic mixed precision (AMP)

```python
# In train.py, before creating optimizer
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

**Expected Speedup**: 1.5-2x on Tesla T4  
**Complexity**: Low (single line change)

#### 3. Optimize Batch Size
**Current**: 256  
**Recommendation**: Experiment with different sizes

Try: 256, 512, 2048, 4096
- Smaller batches: Less memory, possibly faster per-batch
- Larger batches: Better GPU utilization

**Expected Impact**: 10-30% improvement  
**Complexity**: Very Low (just change parameter)

### 🔧 Medium Priority

#### 4. Use `tf.function` Compilation
Mark the training step with `@tf.function` for graph optimization.

**Implementation**:
```python
@tf.function
def train_step(model, optimizer, batch_size, snr_db):
    # ... existing code
```

**Expected Speedup**: 10-20%  
**Complexity**: Low (decorator)

#### 5. Reduce Checkpoint I/O
**Current**: Saves every 10 epochs + best model  
**Impact**: Minimal unless checkpointing is slow

### 📊 Profiling Recommendations

To identify exact bottlenecks, profile the code:

```python
# Add to train.py
import tensorflow as tf

# Profile one training step
tf.profiler.experimental.start('logdir')
train_step(model, optimizer, config.BATCH_SIZE, config.SNR_TRAIN)
tf.profiler.experimental.stop()

# View in TensorBoard:
# tensorboard --logdir=logdir
```

---

## Quick Wins (Try These First)

### ✅ Step 1: Enable Mixed Precision
Add to `train.py` around line 227:

```python
# Setup device
print("\nDevice Setup:")
print_device_info()
device_string, device_name = setup_device(verbose=False)

# ENABLE MIXED PRECISION for ~2x speedup on Tesla T4
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("✅ Mixed precision training enabled (float16)\n")
```

### ✅ Step 2: Try Different Batch Sizes
```bash
# Test various batch sizes
python train.py --batch_size 512 --test_mode
python train.py --batch_size 2048 --test_mode
```

### ✅ Step 3: Add `@tf.function` to Training Step
```python
@tf.function(jit_compile=True)  # Enable XLA compilation
def train_step(model, optimizer, batch_size, snr_db):
    # ... existing code
```

---

## Expected Results

| Optimization | Speedup | Effort |
|--------------|---------|--------|
| Mixed Precision | 1.5-2x | Low |
| Optimal Batch Size | 1.1-1.3x | Very Low |
| @tf.function | 1.1-1.2x | Low |
| All Combined | **~2-3x** | **Low-Medium** |

**Realistic target**: Improve from **~1.1 it/s** to **~3-4 it/s** with quick wins.

For further improvements (5-10x), would need to refactor the sequential sensing loop, which is more complex.

---

## Benchmark Reference

Tesla T4 with optimized deep learning workloads typically achieves:
- 65 TFLOPS (float16)
- 8.1 TFLOPS (float32)

Current speed suggests the GPU is underutilized. The optimizations above should significantly improve utilization.
