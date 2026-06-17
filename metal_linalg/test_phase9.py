"""
Phase 9: a real Metal Householder QR kernel — does a custom GPU QR finally beat CPU?

Background: our CholeskyQR path lost (~0.08x) because it leans on torch's native MPS
cholesky/solve_triangular (slow over many tiny matrices). This replaces it with a
dedicated threadgroup Householder kernel — no torch MPS linalg, no convergence
iteration. The honest prediction was ~50/50 and, if a win, modest: QR's CPU baseline
(LAPACK geqrf on AMX) is the cheapest dense factorization there is.

Phase A: correctness vs LAPACK (reconstruction + orthogonality).
Phase B: speed vs CPU batched qr, and vs the old CholeskyQR path. The gate.

Run:  python -m metal_linalg.test_phase9
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from .accel import gpu_qr, householder_qr

_PASS, _FAIL = 0, 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    _PASS += ok
    _FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _fro(x):
    return float(torch.linalg.norm(x.float()))


def _time(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / repeats * 1e3


def main():
    print(f"torch {torch.__version__} | Phase 9 Metal Householder QR\n")
    rng = np.random.default_rng(0)

    print("Phase A) Correctness vs LAPACK")
    for (m, n) in ((16, 8), (48, 16), (64, 32), (32, 32)):
        A = rng.standard_normal((64, m, n)).astype("f4")
        Q, R = householder_qr(torch.from_numpy(A).to("mps"))
        Am = torch.from_numpy(A).to("mps")
        recon = _fro(Q @ R - Am) / _fro(Am)
        I = torch.eye(n, device="mps")
        orth = max(_fro(Q[i].T @ Q[i] - I) for i in range(0, 64, 8))
        uppertri = float(torch.linalg.norm(torch.tril(R, -1).float()))   # R must be upper
        check(f"{m}x{n}: A=QR, QᵀQ=I, R upper", recon < 1e-4 and orth < 1e-4
              and uppertri < 1e-3, f"recon {recon:.1e} orth {orth:.1e} tril(R) {uppertri:.1e}")

    print("\nPhase B) Speed — the gate: GPU Householder vs CPU vs old CholeskyQR")
    print(f"  {'shape':>9} {'batch':>7} {'HH-GPU':>9} {'CPU':>9} {'CholQR':>9} "
          f"{'HH/CPU':>8} {'HH/CholQR':>10}")
    best = 0.0
    for (m, n) in ((48, 16), (64, 32)):
        for B in (4096, 16384):
            A = rng.standard_normal((B, m, n)).astype("f4")
            Am, Ac = torch.from_numpy(A).to("mps"), torch.from_numpy(A)
            hh = _time(lambda: householder_qr(Am))
            cpu = _time(lambda: torch.linalg.qr(Ac))
            chol = _time(lambda: gpu_qr(Am))
            sp = cpu / hh
            best = max(best, sp)
            print(f"  {m}x{n:>5} {B:>7} {hh:>9.1f} {cpu:>9.1f} {chol:>9.1f} "
                  f"{sp:>7.2f}x {chol / hh:>9.2f}x")
    verdict = "GPU WINS" if best > 1.0 else "CPU still wins"
    print(f"\n  Gate: best HH-GPU vs CPU = {best:.2f}x  ->  {verdict}")

    print(f"\n{'=' * 56}\n{_PASS} passed, {_FAIL} failed (correctness)")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
