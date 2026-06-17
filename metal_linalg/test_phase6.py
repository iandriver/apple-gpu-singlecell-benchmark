"""
Phase 6: parallel-ordering (round-robin) Jacobi eigh vs the sequential kernel.

  1. Correctness of ordering="par" vs LAPACK (the new kernel must be right).
  2. Speed: par vs seq vs CPU across n. Parallel ordering cuts barriers per sweep
     from O(n²) to O(n); the question is how much that buys, and where.

Run:  python -m metal_linalg.test_phase6
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from . import reference as ref
from .kernels import batched_eigh

_PASS, _FAIL = 0, 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    _PASS += ok
    _FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def test_correctness():
    print("1) Parallel-ordering correctness vs LAPACK")
    rng = np.random.default_rng(0)
    for n in (8, 16, 32, 48):
        A = rng.standard_normal((64, n, n)).astype("f4"); A = (A + A.transpose(0, 2, 1)) / 2
        w, V = batched_eigh(torch.from_numpy(A), ordering="par")
        w, V = w.cpu().numpy(), V.cpu().numpy()
        recon = max(ref.recon_error_eigh(A[i], w[i], V[i]) for i in range(len(A)))
        orth = max(ref.orthogonality_error(V[i]) for i in range(len(A)))
        vals = max(ref.values_rel_error(w[i], np.linalg.eigvalsh(A[i].astype(np.float64)))
                   for i in range(len(A)))
        check(f"par n={n}", recon < 1e-4 and orth < 1e-4 and vals < 1e-4,
              f"recon {recon:.1e}  orth {orth:.1e}  vals {vals:.1e}")


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
    print("\n2) Speed: parallel vs sequential vs CPU (batch=16384)")
    print(f"  {'n':>4} {'seq ms':>9} {'par ms':>9} {'par/seq':>9} {'CPU ms':>9} {'par vs CPU':>11}")
    rng = np.random.default_rng(1)
    B = 16384
    for n in (8, 16, 32, 48):
        A = rng.standard_normal((B, n, n)).astype("f4"); A = (A + A.transpose(0, 2, 1)) / 2
        Ag = torch.from_numpy(A).to("mps")
        Ac = torch.from_numpy(A)
        seq = _time(lambda: batched_eigh(Ag, ordering="seq"))
        par = _time(lambda: batched_eigh(Ag, ordering="par"))
        cpu = _time(lambda: torch.linalg.eigh(Ac))
        print(f"  {n:>4} {seq:>9.1f} {par:>9.1f} {seq / par:>8.2f}x {cpu:>9.1f} {cpu / par:>10.2f}x")


def main():
    print(f"torch {torch.__version__} | Phase 6 parallel-ordering Jacobi\n")
    test_correctness()
    test_speed()
    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
