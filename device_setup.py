"""
Device Setup Utility
Automatically detects and configures the best available hardware:
Priority: CUDA > MPS > CPU
"""

import tensorflow as tf
import os
from tensorflow.python.client import device_lib


def _ensure_xla_libdevice(verbose=True):
    """
    Ensure XLA can locate CUDA libdevice.

    Some TensorFlow builds will JIT-compile simple GPU kernels with XLA. If the
    CUDA toolkit is not installed (or not discoverable), XLA fails with:
      "libdevice not found at ./libdevice.10.bc"

    We mitigate this by creating a tiny "CUDA stub" directory containing
    nvvm/libdevice/libdevice.10.bc (symlinked or copied from an existing package,
    e.g., triton), then setting:
      XLA_FLAGS=--xla_gpu_cuda_data_dir=<stub_root>
    """
    existing = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir=" in existing:
        return

    libdevice_src = None
    try:
        import triton  # type: ignore

        triton_root = os.path.dirname(triton.__file__)
        candidate = os.path.join(
            triton_root, "backends", "nvidia", "lib", "libdevice.10.bc"
        )
        if os.path.exists(candidate):
            libdevice_src = candidate
    except Exception:
        libdevice_src = None

    if libdevice_src is None:
        return

    stub_root = os.path.join(os.path.dirname(__file__), ".cuda_stub")
    libdevice_dir = os.path.join(stub_root, "nvvm", "libdevice")
    os.makedirs(libdevice_dir, exist_ok=True)

    dst = os.path.join(libdevice_dir, "libdevice.10.bc")
    if not os.path.exists(dst):
        try:
            os.symlink(libdevice_src, dst)
        except Exception:
            import shutil

            shutil.copyfile(libdevice_src, dst)

    flag = f"--xla_gpu_cuda_data_dir={stub_root}"
    os.environ["XLA_FLAGS"] = (existing + (" " if existing else "") + flag).strip()
    if verbose:
        print(f"✓ XLA libdevice configured via XLA_FLAGS ({stub_root})")


def setup_device(verbose=True):
    """
    Sets up the best available device for TensorFlow operations.

    Priority:
    1. CUDA GPU (NVIDIA)
    2. MPS (Apple Silicon)
    3. CPU

    Returns:
        tuple: (device_string, device_name)
    """
    # Some TensorFlow installs enable XLA auto-jit by default, which can fail on
    # systems missing CUDA "libdevice" (common on partial CUDA setups). We do not
    # rely on XLA for correctness, so disable it for stability.
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

    _ensure_xla_libdevice(verbose=verbose)

    device_name = "CPU"
    device_string = "/CPU:0"

    # CRITICAL: Configure GPU memory growth BEFORE any GPU operations
    # This must be done before ANY TensorFlow operation that would initialize the GPU
    physical_gpus = tf.config.list_physical_devices("GPU")
    if physical_gpus:
        try:
            # Enable memory growth to avoid OOM errors
            # MUST be set before GPU is initialized
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if verbose:
                print(f"✓ Configured memory growth for {len(physical_gpus)} GPU(s)")
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f"⚠ GPU memory configuration error: {e}")
                print(f"  (GPU may have been initialized already)")

    # Check for Apple MPS first (macOS with Apple Silicon)
    try:
        import platform

        if platform.system() == "Darwin":  # macOS
            # Try to set up MPS
            try:
                # Attempt to create a small tensor on MPS to test if it works
                # Only attempt if TF reports a GPU device.
                with tf.device("/GPU:0"):
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    _ = test_tensor + test_tensor

                # If we get here, MPS is available
                device_name = "MPS"
                device_string = "/GPU:0"
                if verbose:
                    print(f"✓ Apple MPS (Metal Performance Shaders) detected")
                    print(f"  Running on Apple Silicon with GPU acceleration")
                return device_string, device_name
            except (RuntimeError, ValueError) as e:
                # MPS not available or error
                if verbose and "MPS" in str(e):
                    print(f"  MPS device found but not functional: {e}")
    except Exception as e:
        if verbose:
            print(f"  MPS detection error: {e}")

    # Check for CUDA GPU (NVIDIA)
    # physical_gpus was already retrieved above before any initialization
    if physical_gpus and device_name == "CPU":
        try:
            # Check if it's a CUDA GPU
            try:
                gpu_details = tf.config.experimental.get_device_details(physical_gpus[0])
                if "device_name" in gpu_details:
                    device_name = "CUDA GPU"
                    device_string = "/GPU:0"
                    if verbose:
                        print(
                            f"✓ CUDA GPU detected: {gpu_details.get('device_name', 'Unknown')}"
                        )
                        print(f"  Number of GPUs: {len(physical_gpus)}")
            except:
                # Might be MPS showing up as GPU
                device_name = "CUDA GPU"
                device_string = "/GPU:0"
                if verbose:
                    print(f"✓ GPU detected")
                    print(f"  Number of GPUs: {len(physical_gpus)}")
        except RuntimeError as e:
            if verbose:
                print(f"GPU configuration error: {e}")

    # Fallback to CPU
    if device_name == "CPU":
        if verbose:
            print(f"✓ Using CPU")
            print(f"  No GPU acceleration available")

    if verbose:
        print(f"\n→ Selected device: {device_name}")
        print(f"→ Device string: {device_string}\n")

    return device_string, device_name


def get_device_strategy():
    """
    Returns appropriate TensorFlow distribution strategy based on available hardware.

    Returns:
        tf.distribute.Strategy: Distribution strategy for training
    """
    device_string, device_name = setup_device(verbose=False)

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) > 1:
        # Multiple GPUs available - use MirroredStrategy
        print(f"Using MirroredStrategy with {len(gpus)} GPUs")
        strategy = tf.distribute.MirroredStrategy()
    else:
        # Single device (GPU or CPU) - use default strategy
        strategy = tf.distribute.get_strategy()

    return strategy


def print_device_info():
    """
    Prints detailed information about available devices.
    """
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)

    # TensorFlow version
    print(f"\nTensorFlow Version: {tf.__version__}")

    # CPU info
    print(f"\nCPU Devices: {len(tf.config.list_physical_devices('CPU'))}")

    # GPU info
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU Devices: {len(gpus)}")

    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"\n  GPU {i}: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                for key, value in details.items():
                    print(f"    {key}: {value}")
            except:
                print(f"    (Details not available)")

    # MPS info (Apple Silicon)
    try:
        mps_devices = tf.config.list_physical_devices("MPS")
        if mps_devices:
            print(f"\nMPS Devices: {len(mps_devices)}")
    except:
        pass

    print("\n" + "=" * 60)

    # Setup and show selected device
    print("\n")
    setup_device(verbose=True)


if __name__ == "__main__":
    print_device_info()
