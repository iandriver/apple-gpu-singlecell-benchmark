"""
Phase 3: batched one-sided Jacobi SVD.

  1. Correctness vs LAPACK: reconstruction ‖A − U·diag(S)·Vh‖, U/V orthonormality,
     singular values — for tall AND wide batches. (Singular vectors carry sign/order
     ambiguity, so we grade invariants, not raw vectors.)
  2. Speed: GPU batched_svd vs CPU torch.linalg.svd (batched LAPACK).

Run:  python -m metal_linalg.test_phase3
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from . import reference as ref
from .kernels import batched_svd

_PASS, _FAIL = 0, 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    _PASS += ok
    _FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def test_correctness():
    print("1) Correctness vs LAPACK (per-matrix), batched one-sided Jacobi SVD")
    rng = np.random.default_rng(0)
    for tag, (m, n) in {"tall 48x16": (48, 16), "square 32x32": (32, 32),
                        "tall 64x32": (64, 32), "wide 16x48": (16, 48)}.items():
        A = rng.standard_normal((64, m, n)).astype("f4")
        U, S, Vh = batched_svd(torch.from_numpy(A))
        U, S, Vh = U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy()
        recon = max(ref.recon_error_svd(A[i], U[i], S[i], Vh[i]) for i in range(len(A)))
        orthU = max(ref.orthogonality_error(U[i]) for i in range(len(A)))
        orthV = max(ref.orthogonality_error(Vh[i].T) for i in range(len(A)))
        vals = max(ref.values_rel_error(S[i], np.linalg.svd(A[i].astype(np.float64),
                   compute_uv=False)) for i in range(len(A)))
        ok = recon < 1e-4 and orthU < 1e-4 and orthV < 1e-4 and vals < 1e-4
        check(tag, ok, f"recon {recon:.1e}  U⊥ {orthU:.1e}  V⊥ {orthV:.1e}  S {vals:.1e}")


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
    print("\n2) Speed: GPU batched SVD vs CPU/Accelerate (batched LAPACK)")
    print(f"  {'shape':>10} {'batch':>7} {'GPU ms':>9} {'CPU ms':>9} {'speedup':>9}")
    best = 0.0
    rng = np.random.default_rng(1)
    for (m, n) in ((48, 16), (64, 32)):
        for B in (1024, 4096, 16384):
            A = rng.standard_normal((B, m, n)).astype("f4")
            Ag = torch.from_numpy(A).to("mps")
            Ac = torch.from_numpy(A)
            gpu = _time(lambda: batched_svd(Ag))
            cpu = _time(lambda: torch.linalg.svd(Ac, full_matrices=False))
            sp = cpu / gpu
            best = max(best, sp)
            print(f"  {m}x{n:>6} {B:>7} {gpu:>9.1f} {cpu:>9.1f} {sp:>8.2f}x"
                  + ("  <-- GPU wins" if sp > 1 else ""))
    check("GPU achieves a real SVD speedup", best > 1.0, f"best {best:.2f}x")


def main():
    print(f"torch {torch.__version__} | Phase 3 batched one-sided Jacobi SVD\n")
    test_correctness()
    test_speed()
    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
