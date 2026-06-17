"""
Phase 10: fused Householder QR-solve makes lstsq a GPU win.

lstsq via QR + torch.linalg.solve_triangular lost (~0.24x) because the triangular
solve fell back to torch's slow MPS path. Fusing Qᵀb + back-substitution into the
kernel keeps the whole solve on-GPU. Correctness vs CPU + speed.

Run:  python -m metal_linalg.test_phase10
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


def main():
    print(f"torch {torch.__version__} | Phase 10 fused QR-solve lstsq\n")
    metal_linalg.install()
    rng = np.random.default_rng(0)

    print("1) Correctness vs CPU LAPACK (matrix-RHS and vector-RHS)")
    for (m, n, tag) in ((48, 16, "matrix-RHS k=4"), (64, 32, "vector-RHS")):
        A = rng.standard_normal((64, m, n)).astype("f4")
        if "vector" in tag:
            B = rng.standard_normal((64, m)).astype("f4")
        else:
            B = rng.standard_normal((64, m, 4)).astype("f4")
        sol = torch.linalg.lstsq(torch.from_numpy(A).to("mps"),
                                 torch.from_numpy(B).to("mps")).solution
        solc = torch.linalg.lstsq(torch.from_numpy(A), torch.from_numpy(B)).solution
        err = float(torch.linalg.norm((sol.cpu() - solc).float())
                    / torch.linalg.norm(solc.float()))
        check(f"{m}x{n} {tag}: matches CPU, mps out, shape {tuple(sol.shape)}",
              err < 1e-3 and sol.device.type == "mps", f"err {err:.1e}")

    print("\n2) Speed: GPU fused lstsq vs CPU/Accelerate")
    print(f"  {'shape':>9} {'batch':>7} {'GPU ms':>9} {'CPU ms':>9} {'speedup':>9}")
    best = 0.0
    for (m, n) in ((48, 16), (64, 32)):
        for Bn in (4096, 16384):
            A = rng.standard_normal((Bn, m, n)).astype("f4")
            bb = rng.standard_normal((Bn, m, 1)).astype("f4")
            Am, Ac = torch.from_numpy(A).to("mps"), torch.from_numpy(A)
            Bm, Bc = torch.from_numpy(bb).to("mps"), torch.from_numpy(bb)
            g = _time(lambda: torch.linalg.lstsq(Am, Bm))
            c = _time(lambda: torch.linalg.lstsq(Ac, Bc))
            best = max(best, c / g)
            print(f"  {m}x{n:>5} {Bn:>7} {g:>9.1f} {c:>9.1f} {c / g:>8.2f}x"
                  + ("  <-- GPU wins" if c > g else ""))
    check("lstsq is a real GPU win now", best > 1.0, f"best {best:.2f}x")

    metal_linalg.uninstall()
    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
