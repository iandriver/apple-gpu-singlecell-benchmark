"""
Phase 5: drop-in integration test.

  1. Before install: torch.linalg.eigh on an MPS tensor fails (the gap we fill).
  2. After install: torch.linalg.eigh / svd on MPS tensors work, are correct, and
     hit the fast batched kernel.
  3. CPU path is untouched (still the original LAPACK).
  4. uninstall() restores the originals.

Run:  python -m metal_linalg.test_phase5
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import metal_linalg
from . import reference as ref

_PASS, _FAIL = 0, 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    _PASS += ok
    _FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    print(f"torch {torch.__version__} | Phase 5 drop-in torch.linalg patch\n")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((128, 24, 24)).astype("f4"); A = (A + A.transpose(0, 2, 1)) / 2
    Am = torch.from_numpy(A).to("mps")

    print("1) Before install: MPS eigh is unsupported by stock torch")
    try:
        torch.linalg.eigh(Am)
        check("stock torch.linalg.eigh(mps) raises", False, "did not raise")
    except Exception as e:
        check("stock torch.linalg.eigh(mps) raises (the gap)", True, type(e).__name__)

    print("2) After install: torch.linalg.eigh / svd work on MPS")
    metal_linalg.install()
    w, V = torch.linalg.eigh(Am)
    check("eigh returns mps tensors", w.device.type == "mps" and V.device.type == "mps")
    e = max(ref.recon_error_eigh(A[i], w[i].cpu().numpy(), V[i].cpu().numpy())
            for i in range(0, 128, 8))
    check("patched eigh correct", e < 1e-4, f"recon {e:.1e}")

    Bsv = rng.standard_normal((64, 40, 20)).astype("f4")
    U, S, Vh = torch.linalg.svd(torch.from_numpy(Bsv).to("mps"), full_matrices=False)
    e2 = max(ref.recon_error_svd(Bsv[i], U[i].cpu().numpy(), S[i].cpu().numpy(),
             Vh[i].cpu().numpy()) for i in range(0, 64, 8))
    check("patched svd correct", e2 < 1e-4 and U.device.type == "mps", f"recon {e2:.1e}")

    print("3) CPU path untouched (still original LAPACK)")
    Ac = torch.from_numpy(A[0])
    w_cpu, V_cpu = torch.linalg.eigh(Ac)
    ref_w = np.linalg.eigvalsh(A[0].astype(np.float64))
    check("cpu eigh still correct", ref.values_rel_error(w_cpu.numpy(), ref_w) < 1e-5)

    print("4) uninstall restores originals")
    metal_linalg.uninstall()
    try:
        torch.linalg.eigh(Am)
        check("eigh(mps) raises again after uninstall", False)
    except Exception:
        check("eigh(mps) raises again after uninstall", True)

    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
