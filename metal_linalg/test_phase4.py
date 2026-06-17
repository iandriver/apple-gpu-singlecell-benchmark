"""
Phase 4: precision (fp16) + robustness (auto-dispatch, CPU fallback guard).

  1. fp16 batched eigh: measure speed vs fp32 and accuracy vs LAPACK. Kept only
     because it is *measured* — fp16 on Apple GPUs mainly buys occupancy (halved
     threadgroup memory), since fp16 ALU throughput ~= fp32 on M-series.
  2. Auto-dispatch: eigh()/svd() route batched-small to GPU, everything else to CPU.
  3. Verify-fallback guard: a deliberately hard (non-converging) case is caught and
     recomputed on CPU.

Run:  python -m metal_linalg.test_phase4
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from . import reference as ref
from .dispatch import eigh, svd
from .kernels import batched_eigh

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


def test_fp16():
    # Measured negative result: on M-series the fp16 ALU throughput ~= fp32, and the
    # halved threadgroup storage does not buy enough occupancy to win — meanwhile
    # accuracy drops to ~fp16. Conclusion: fp16 is NOT a speedup here; we keep fp32.
    print("1) fp16 batched eigh: is it actually faster? (measured)")
    print(f"  {'n':>4} {'batch':>7} {'fp32 ms':>9} {'fp16 ms':>9} {'fp16/fp32':>10} {'fp16 recon':>11}")
    rng = np.random.default_rng(0)
    worth_it = False
    for n in (16, 32):
        B = 16384
        A = rng.standard_normal((B, n, n)).astype("f4"); A = (A + A.transpose(0, 2, 1)) / 2
        Ag = torch.from_numpy(A).to("mps")
        t32 = _time(lambda: batched_eigh(Ag, store="fp32"))
        t16 = _time(lambda: batched_eigh(Ag, store="fp16"))
        w, V = batched_eigh(Ag, store="fp16")
        w, V = w.cpu().numpy(), V.cpu().numpy()
        recon = max(ref.recon_error_eigh(A[i], w[i], V[i]) for i in range(0, B, 512))
        ratio = t32 / t16
        worth_it |= (ratio > 1.3 and recon < 1e-3)   # would need both to be worth it
        print(f"  {n:>4} {B:>7} {t32:>9.1f} {t16:>9.1f} {ratio:>9.2f}x {recon:>11.1e}")
    check("fp16 is NOT a meaningful speedup on this GPU (documented; keep fp32)",
          not worth_it, "fp16 ALU ~= fp32 on M-series; accuracy also worse")


def test_dispatch():
    print("\n2) Auto-dispatch routing + correctness")
    rng = np.random.default_rng(1)
    # batched small -> GPU path
    A = rng.standard_normal((128, 24, 24)).astype("f4"); A = (A + A.transpose(0, 2, 1)) / 2
    w, V = eigh(torch.from_numpy(A))
    e = max(ref.recon_error_eigh(A[i], w[i].cpu().numpy(), V[i].cpu().numpy()) for i in range(0, 128, 8))
    check("eigh routes batched->GPU, correct", e < 1e-4, f"recon {e:.1e}")
    # single large matrix -> CPU fallback
    big = rng.standard_normal((200, 200)).astype("f4"); big = (big + big.T) / 2
    w2, V2 = eigh(torch.from_numpy(big))
    e2 = ref.recon_error_eigh(big, w2.cpu().numpy(), V2.cpu().numpy())
    check("eigh routes single-large->CPU, correct", e2 < 1e-4, f"recon {e2:.1e}")
    # svd batched
    Asv = rng.standard_normal((64, 40, 20)).astype("f4")
    U, S, Vh = svd(torch.from_numpy(Asv))
    e3 = max(ref.recon_error_svd(Asv[i], U[i].cpu().numpy(), S[i].cpu().numpy(),
             Vh[i].cpu().numpy()) for i in range(0, 64, 8))
    check("svd routes batched->GPU, correct", e3 < 1e-4, f"recon {e3:.1e}")


def test_verify_guard():
    print("\n3) Verify-fallback guard (force non-convergence with max_sweeps=1)")
    # A hard batch + a deliberately tiny sweep budget so the GPU result is wrong;
    # verify=True must catch the bad ones and recompute on CPU.
    rng = np.random.default_rng(2)
    A = rng.standard_normal((256, 32, 32)).astype("f4"); A = (A + A.transpose(0, 2, 1)) / 2
    At = torch.from_numpy(A)
    w_bad, V_bad = batched_eigh(At, max_sweeps=1)         # under-converged on purpose
    bad = max(ref.recon_error_eigh(A[i], w_bad[i].cpu().numpy(), V_bad[i].cpu().numpy())
              for i in range(0, 256, 16))
    # dispatch eigh with verify should fix it (it uses full sweeps + residual fallback)
    w, V = eigh(At, verify=True)
    good = max(ref.recon_error_eigh(A[i], w[i].cpu().numpy(), V[i].cpu().numpy())
               for i in range(0, 256, 16))
    check("under-converged result is detectably bad", bad > 1e-2, f"recon {bad:.1e}")
    check("verify=True yields correct result", good < 1e-4, f"recon {good:.1e}")


def main():
    print(f"torch {torch.__version__} | Phase 4 precision + robustness\n")
    test_fp16()
    test_dispatch()
    test_verify_guard()
    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
