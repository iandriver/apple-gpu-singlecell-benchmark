"""
Phase 8: push acceleration. Finding: a GPU op only WINS when it routes through our
custom Metal kernel (batched SVD/eigh) — not through torch's native MPS cholesky/
solve_triangular, which is slow for batched-tiny. So:
  * pinv / lstsq / matrix_rank -> our GPU SVD  (WIN)
  * qr -> CPU round-trip, because GPU CholeskyQR measured ~12x SLOWER (shown below)

Correctness vs CPU LAPACK + GPU-vs-CPU speed.

Run:  python -m metal_linalg.test_phase8
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

import metal_linalg

_PASS, _FAIL = 0, 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    _PASS += ok
    _FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _time(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / repeats * 1e3


def _fro(x):
    return float(torch.linalg.norm(x.float()))


def main():
    print(f"torch {torch.__version__} | Phase 8 accelerate qr/pinv/lstsq/matrix_rank\n")
    metal_linalg.install()
    rng = np.random.default_rng(0)
    B = 8192

    print("1) qr — works on MPS (CPU round-trip; GPU CholeskyQR was slower)")
    A = rng.standard_normal((B, 48, 32)).astype("f4")
    Am = torch.from_numpy(A).to("mps")
    Q, R = torch.linalg.qr(Am)            # patched -> CPU round-trip, mps out
    recon = _fro(Q @ R - Am) / _fro(Am)
    check("qr: A=QR correct, mps out (via round-trip)", recon < 1e-4
          and Q.device.type == "mps", f"recon {recon:.1e}")
    from metal_linalg.accel import gpu_qr
    g = _time(lambda: gpu_qr(Am)); c = _time(lambda: torch.linalg.qr(torch.from_numpy(A)))
    print(f"     (GPU CholeskyQR {g:.1f} ms vs CPU {c:.1f} ms -> {c / g:.2f}x: "
          f"GPU loses, so qr stays CPU round-trip)")

    print("\n2) pinv (GPU SVD) — correctness + speed")
    A = rng.standard_normal((B, 40, 20)).astype("f4")
    Am, Ac = torch.from_numpy(A).to("mps"), torch.from_numpy(A)
    P = torch.linalg.pinv(Am)
    err = _fro(Am @ P @ Am - Am) / _fro(Am)
    check("pinv: A P A == A, on MPS", err < 1e-3 and P.device.type == "mps", f"{err:.1e}")
    g = _time(lambda: torch.linalg.pinv(Am)); c = _time(lambda: torch.linalg.pinv(Ac))
    print(f"     GPU {g:.1f} ms  vs  CPU {c:.1f} ms  ->  {c / g:.2f}x")

    print("\n3) lstsq — works on MPS (CPU round-trip; GPU-SVD path was slower)")
    A = rng.standard_normal((B, 40, 24)).astype("f4")
    Bb = rng.standard_normal((B, 40, 1)).astype("f4")
    Am, Bm = torch.from_numpy(A).to("mps"), torch.from_numpy(Bb).to("mps")
    sol = torch.linalg.lstsq(Am, Bm).solution            # patched -> CPU round-trip
    sol_cpu = torch.linalg.lstsq(torch.from_numpy(A), torch.from_numpy(Bb)).solution
    err = _fro(sol.cpu() - sol_cpu) / _fro(sol_cpu)
    check("lstsq: correct, mps out (via round-trip)", err < 1e-3 and sol.device.type == "mps",
          f"{err:.1e}")
    from metal_linalg.accel import gpu_lstsq
    g = _time(lambda: gpu_lstsq(Am, Bm))
    c = _time(lambda: torch.linalg.lstsq(torch.from_numpy(A), torch.from_numpy(Bb)))
    print(f"     (GPU-SVD lstsq {g:.1f} ms vs CPU {c:.1f} ms -> {c / g:.2f}x: CPU lstsq "
          f"uses cheap QR, so GPU loses -> round-trip)")

    print("\n4) matrix_rank (GPU svdvals) — correctness + speed")
    A = rng.standard_normal((B, 24, 16)).astype("f4")     # full rank -> 16
    Am, Ac = torch.from_numpy(A).to("mps"), torch.from_numpy(A)
    r = torch.linalg.matrix_rank(Am)
    check("matrix_rank: == 16, on MPS", bool((r == 16).all()) and r.device.type == "mps",
          f"min {int(r.min())} max {int(r.max())}")
    g = _time(lambda: torch.linalg.matrix_rank(Am))
    c = _time(lambda: torch.linalg.matrix_rank(Ac))
    print(f"     GPU {g:.1f} ms  vs  CPU {c:.1f} ms  ->  {c / g:.2f}x")

    metal_linalg.uninstall()
    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
