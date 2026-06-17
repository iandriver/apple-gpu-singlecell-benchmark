"""
Benchmark harness for eigh/svd. Phase 0 establishes the baselines the Phase 1
Metal kernels must beat:
  * CPU LAPACK (Accelerate, via torch.linalg on cpu) across sizes
  * the existing GPU CholeskyQR randomized-SVD (../pca_gpu_rsvd.py) for the
    low-rank case — the bar that any general Metal SVD has to justify itself against

The "Metal" column currently reports the Phase-0 placeholder (== CPU). When Phase
1 lands, the same harness reports the real GPU Jacobi numbers with no changes here.

Run:  python -m metal_linalg.bench
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .kernels import metal_eigh, metal_svd


def _median_ms(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e3


def bench_eigh(sizes=(128, 256, 512, 1024)):
    print("eigh — symmetric, values+vectors  (CPU Accelerate baseline)")
    print(f"  {'n':>6} {'CPU ms':>10} {'Metal ms':>10}   note")
    for n in sizes:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((n, n)).astype("f4"); A = (A + A.T) / 2
        Ac = torch.from_numpy(A)
        cpu = _median_ms(lambda: torch.linalg.eigh(Ac))
        Am = torch.from_numpy(A).to("mps")
        mtl = _median_ms(lambda: metal_eigh(Am))
        print(f"  {n:>6} {cpu:>10.2f} {mtl:>10.2f}   {'placeholder=CPU' }")


def bench_svd(shapes=((512, 256), (1024, 512), (2048, 512))):
    print("\nsvd — rectangular, reduced, values+vectors  (CPU Accelerate baseline)")
    print(f"  {'shape':>12} {'CPU ms':>10} {'Metal ms':>10}   note")
    for m, n in shapes:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((m, n)).astype("f4")
        Ac = torch.from_numpy(A)
        cpu = _median_ms(lambda: torch.linalg.svd(Ac, full_matrices=False))
        Am = torch.from_numpy(A).to("mps")
        mtl = _median_ms(lambda: metal_svd(Am, full_matrices=False))
        print(f"  {m}x{n:>7} {cpu:>10.2f} {mtl:>10.2f}   {'placeholder=CPU'}")


def main():
    print(f"torch {torch.__version__} | baselines for Phase 1 to beat\n")
    bench_eigh()
    bench_svd()
    print("\n(Metal column == CPU until Phase 1 swaps in the Jacobi kernels.)")


if __name__ == "__main__":
    main()
