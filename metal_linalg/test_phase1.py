"""
Phase 1 acceptance test: GPU two-sided Jacobi symmetric eigh.

The go/no-go bar is CORRECTNESS, graded by the Phase-0 harness against LAPACK on
the pathological matrices:
  * reconstruction   ‖A − V·diag(w)·Vᵀ‖ / ‖A‖
  * orthogonality    ‖VᵀV − I‖
  * eigenvalues      max|w − w_lapack| / max|w_lapack|

(We do NOT compare individual eigenvectors for clustered/degenerate spectra — those
directions are genuinely ambiguous; reconstruction + orthogonality are the right
invariants there.)

Speed is reported but is NOT the bar: a single threadgroup uses a sliver of the GPU,
so CPU/AMX wins at these small sizes. Phase 2 addresses speed and large n.

Run:  python -m metal_linalg.test_phase1
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from . import reference as ref
from .kernels import metal_eigh

_PASS, _FAIL = 0, 0
TOL = 1e-4   # generous fp32-Jacobi bar; raw numbers printed so regressions show


def check(name, ok, detail=""):
    global _PASS, _FAIL
    _PASS += ok
    _FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def grade(name, A):
    """Run metal_eigh on A and grade against LAPACK."""
    w, V = metal_eigh(torch.from_numpy(A))
    w, V = w.cpu().numpy(), V.cpu().numpy()
    w_ref = np.linalg.eigvalsh(A.astype(np.float64))
    recon = ref.recon_error_eigh(A, w, V)
    orth = ref.orthogonality_error(V)
    vals = ref.values_rel_error(w, w_ref)
    ok = recon < TOL and orth < TOL and vals < TOL
    check(name, ok, f"recon {recon:.1e}  orth {orth:.1e}  vals {vals:.1e}")
    return ok


def test_pathological():
    print("1) Pathological symmetric matrices (n=64), GPU Jacobi vs LAPACK")
    for name, A in ref.symmetric_cases(n=64).items():
        grade(name, A)


def test_sizes():
    print("2) Correctness across sizes (random_sym)")
    rng = np.random.default_rng(7)
    for n in (8, 16, 32, 64, 128, 256):
        A = rng.standard_normal((n, n)).astype("f4"); A = (A + A.T) / 2
        grade(f"n={n:<4}", A)


def test_timing():
    print("3) Timing (single-threadgroup GPU vs CPU/Accelerate) — speed is NOT the bar")
    print(f"  {'n':>5} {'GPU ms':>9} {'CPU ms':>9}")
    rng = np.random.default_rng(0)
    for n in (32, 64, 128, 256):
        A = rng.standard_normal((n, n)).astype("f4"); A = (A + A.T) / 2
        At = torch.from_numpy(A)
        # warm + time GPU
        metal_eigh(At); torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(3):
            metal_eigh(At); torch.mps.synchronize()
        gpu = (time.perf_counter() - t) / 3 * 1e3
        t = time.perf_counter()
        for _ in range(3):
            torch.linalg.eigh(At)
        cpu = (time.perf_counter() - t) / 3 * 1e3
        print(f"  {n:>5} {gpu:>9.1f} {cpu:>9.2f}")


def main():
    print(f"torch {torch.__version__} | Phase 1 GPU Jacobi eigh\n")
    test_pathological()
    test_sizes()
    test_timing()
    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
