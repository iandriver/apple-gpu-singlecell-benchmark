"""
Probe: which linear-algebra routines actually run on the Apple GPU via
PyTorch's MPS backend?

This is the crux of the whole benchmark. PCA, neighbor graphs, and most of the
expensive single-cell steps reduce to SVD / QR / eigendecomposition. If those
don't run on the GPU, there is no point porting the pipeline to Metal.

Each routine is guarded by a SIGALRM watchdog because some MPS linalg kernels
*hang indefinitely* rather than erroring.

Run:  python probe_mps_linalg.py
Expected (PyTorch 2.12, macOS / Apple Silicon):
    pca_lowrank q=50       TIMEOUT  (uses tall-skinny QR internally)
    linalg.qr (tall)       TIMEOUT
    linalg.svd full        OK but via SILENT CPU FALLBACK (UserWarning)
    randomized-svd matmul  TIMEOUT  (the QR step)
    gram + eigh 2000       ERROR: aten::_linalg_eigh not implemented on MPS
"""

import signal
import time

import numpy as np
import torch


class _Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_Timeout()))

DEV = torch.device("mps")
torch.manual_seed(0)
X = torch.randn(50_000, 2_000, device=DEV)
Xc = X - X.mean(0, keepdim=True)


def trial(name, fn, limit=30):
    signal.alarm(limit)
    t = time.perf_counter()
    try:
        fn()
        torch.mps.synchronize()
        print(f"{name:<24} OK   {(time.perf_counter() - t) * 1e3:.1f} ms", flush=True)
    except _Timeout:
        print(f"{name:<24} TIMEOUT >{limit}s (hang)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name:<24} ERROR {repr(e)[:90]}", flush=True)
    finally:
        signal.alarm(0)


def randomized_svd():
    g = torch.randn(2_000, 60, device=DEV)
    Y = Xc @ g
    Q, _ = torch.linalg.qr(Y)        # <- tall-skinny QR, the hang
    B = Q.t() @ Xc
    Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :50] * S[:50]


def gram_eigh():
    C = Xc.t() @ Xc                  # GPU matmul is fine
    w, v = torch.linalg.eigh(C)      # <- not implemented on MPS
    return Xc @ v[:, -50:]


if __name__ == "__main__":
    print(f"torch {torch.__version__} | device {DEV}")
    trial("pca_lowrank q=50", lambda: torch.pca_lowrank(Xc, q=50, niter=4))
    trial("linalg.qr (tall)", lambda: torch.linalg.qr(Xc[:, :50]))
    trial("linalg.svd full", lambda: torch.linalg.svd(Xc, full_matrices=False))
    trial("randomized-svd matmul", randomized_svd)
    trial("gram + eigh 2000", gram_eigh)
