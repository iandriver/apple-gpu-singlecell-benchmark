"""
Phase 2: batched Jacobi eigh — the actual GPU-speedup test.

Two things, in order:
  1. Correctness: every matrix in the batch must match LAPACK (reconstruction,
     orthogonality, eigenvalues).
  2. Speed: GPU batched_eigh vs CPU torch.linalg.eigh (itself batched LAPACK)
     across batch sizes and n. We report real numbers and the crossover where the
     GPU starts winning. (MPS has no eigh, so CPU is the only baseline.)

Run:  python -m metal_linalg.test_phase2
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from . import reference as ref
from .kernels import BATCH_N_MAX, batched_eigh

_PASS, _FAIL = 0, 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    _PASS += ok
    _FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _make_batch(B, n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((B, n, n)).astype("f4")
    A = (A + A.transpose(0, 2, 1)) / 2
    return A


def test_correctness():
    print("1) Correctness vs LAPACK (per-matrix), batched GPU Jacobi")
    for n in (8, 16, 32):
        A = _make_batch(64, n, seed=n)
        w, V = batched_eigh(torch.from_numpy(A))
        w, V = w.cpu().numpy(), V.cpu().numpy()
        recon = max(ref.recon_error_eigh(A[i], w[i], V[i]) for i in range(len(A)))
        orth = max(ref.orthogonality_error(V[i]) for i in range(len(A)))
        vals = max(ref.values_rel_error(w[i], np.linalg.eigvalsh(A[i].astype(np.float64)))
                   for i in range(len(A)))
        check(f"n={n:<3} batch=64", recon < 1e-4 and orth < 1e-4 and vals < 1e-4,
              f"max recon {recon:.1e}  orth {orth:.1e}  vals {vals:.1e}")


def _time(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / repeats * 1e3


def test_speed():
    print("\n2) Speed: GPU batched vs CPU/Accelerate (batched LAPACK)")
    print(f"  {'n':>4} {'batch':>7} {'GPU ms':>9} {'CPU ms':>9} {'speedup':>9}")
    best = 0.0
    for n in (16, 32):
        for B in (256, 1024, 4096, 16384):
            A = _make_batch(B, n, seed=1)
            Ag = torch.from_numpy(A).to("mps")
            Ac = torch.from_numpy(A)  # cpu
            gpu = _time(lambda: batched_eigh(Ag))
            cpu = _time(lambda: torch.linalg.eigh(Ac))
            sp = cpu / gpu
            best = max(best, sp)
            flag = "  <-- GPU wins" if sp > 1 else ""
            print(f"  {n:>4} {B:>7} {gpu:>9.1f} {cpu:>9.1f} {sp:>8.2f}x{flag}")
    check("GPU achieves a real speedup somewhere", best > 1.0,
          f"best observed {best:.2f}x")


def main():
    print(f"torch {torch.__version__} | Phase 2 batched eigh (n <= {BATCH_N_MAX})\n")
    test_correctness()
    test_speed()
    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
