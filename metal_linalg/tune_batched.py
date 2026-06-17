"""
Occupancy autotune for batched_eigh. Sweeps threads-per-matrix (BATCH_BTG) and
the threadgroup footprint (BATCH_MAX_BN) to find the fastest config per n, and
compares against the old baseline (max_bn=32, btg=64) and the CPU.

Run:  python -m metal_linalg.tune_batched
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .kernels import _bucket, batched_eigh


def _time(fn, repeats=4, warmup=2):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / repeats * 1e3


def main():
    B = 8192
    print(f"torch {torch.__version__} | autotune batched_eigh, batch={B}\n")
    for n in (8, 16, 32):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((B, n, n)).astype("f4"); A = (A + A.transpose(0, 2, 1)) / 2
        Ag = torch.from_numpy(A).to("mps")
        Ac = torch.from_numpy(A)
        cpu = _time(lambda: torch.linalg.eigh(Ac))
        base = _time(lambda: batched_eigh(Ag, max_bn=32, btg=64))   # Phase-2 baseline

        print(f"n={n}  CPU {cpu:.1f} ms  |  baseline(max_bn=32,btg=64) "
              f"{base:.1f} ms ({cpu / base:.2f}x)")
        best_ms, best_cfg = base, ("32", "64")
        for btg in (16, 32, 64, 128):
            mb = _bucket(n)
            ms = _time(lambda: batched_eigh(Ag, max_bn=mb, btg=btg))
            mark = ""
            if ms < best_ms:
                best_ms, best_cfg = ms, (mb, btg)
                mark = " *"
            print(f"    max_bn={mb:<2} btg={btg:<3}  {ms:6.1f} ms  {cpu / ms:5.2f}x{mark}")
        print(f"  -> best: max_bn={best_cfg[0]} btg={best_cfg[1]}  "
              f"{best_ms:.1f} ms  {cpu / best_ms:.2f}x  "
              f"(vs baseline {base / best_ms:.2f}x faster)\n")


if __name__ == "__main__":
    main()
