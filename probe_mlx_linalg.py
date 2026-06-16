"""
Probe: does MLX (Apple's own array framework) run SVD / QR / eigh on the GPU?

PyTorch-MPS cannot (see probe_mps_linalg.py). MLX *exposes* mlx.core.linalg.svd,
qr, and eigh, so the hope was that MLX could do the eigendecomposition that PCA
needs on the Apple GPU. This script tests that directly.

Run:  python probe_mlx_linalg.py
Result (MLX 0.31.2): all three raise
    ValueError: [linalg::svd]  This op is not yet supported on the GPU.
                               Explicitly pass a CPU stream.
i.e. MLX linalg is CPU-only too. There is currently NO GPU eigensolver / SVD /
QR on Apple Silicon via either PyTorch-MPS or MLX.
"""

import signal
import time

import mlx.core as mx
import numpy as np


class _Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_Timeout()))

np.random.seed(0)
X = mx.array(np.random.randn(50_000, 2_000).astype("f4"))
Xc = X - mx.mean(X, axis=0, keepdims=True)


def trial(name, fn, limit=60):
    signal.alarm(limit)
    try:
        for _ in range(2):
            mx.eval(fn())            # warmup
        t = time.perf_counter()
        for _ in range(3):
            mx.eval(fn())
        print(f"{name:<26} OK   {(time.perf_counter() - t) / 3 * 1e3:.1f} ms", flush=True)
    except _Timeout:
        print(f"{name:<26} TIMEOUT >{limit}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name:<26} ERROR {repr(e)[:90]}", flush=True)
    finally:
        signal.alarm(0)


def gpu_pca():
    C = Xc.T @ Xc
    w, V = mx.linalg.eigh(C, stream=mx.gpu)
    return Xc @ V[:, -50:]


if __name__ == "__main__":
    print(f"mlx {mx.__version__} | default device {mx.default_device()}")
    trial("svd (50k x 2k)", lambda: mx.linalg.svd(Xc, stream=mx.gpu))
    trial("qr (50k x 2k)", lambda: mx.linalg.qr(Xc, stream=mx.gpu))
    C = Xc.T @ Xc
    mx.eval(C)
    trial("eigh (2k x 2k Gram)", lambda: mx.linalg.eigh(C, stream=mx.gpu))
    trial("full GPU PCA (gram+eigh)", gpu_pca)
