"""
Phase 0 acceptance test. Proves:
  1. torch.mps.compile_shader is available and our Metal source compiles.
  2. A trivial kernel (saxpy) runs on the GPU with correct, zero-copy I/O.
  3. The forward-looking primitive (apply_col_rotation) matches a NumPy Givens
     rotation bit-for-bit (within fp32 tol) — the building block Phase 1 reuses.
  4. The accuracy harness metrics are sound (exact LAPACK decompositions score ~0).
  5. The pathological test-matrix generators produce every case.

Run:  python -m metal_linalg.test_phase0     (from the repo root)
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

from . import reference as ref
from ._dispatch import mps_available
from .kernels import apply_col_rotation, metal_eigh, metal_svd, saxpy

_PASS, _FAIL = 0, 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


def test_integration():
    print("1) Integration / compile_shader")
    check("MPS + compile_shader available", mps_available())

    # saxpy
    x = torch.arange(1000, dtype=torch.float32)
    y = torch.arange(1000, dtype=torch.float32) * 2
    out = saxpy(x, y, a=3.0).cpu().numpy()
    expect = 3.0 * x.numpy() + y.numpy()
    err = float(np.max(np.abs(out - expect)))
    check("saxpy matches NumPy", err < 1e-4, f"max abs err {err:.2e}")


def test_rotation_primitive():
    print("2) Jacobi rotation primitive (Phase-1 building block)")
    n = 64
    rng = np.random.default_rng(1)
    A = rng.standard_normal((n, n)).astype("f4")
    p, q = 3, 50
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)

    # GPU
    gpu = apply_col_rotation(torch.from_numpy(A.copy()), p, q, c, s).cpu().numpy()
    # NumPy reference: A' = A @ G(p,q,theta)
    G = np.eye(n, dtype=np.float64)
    G[p, p] = c; G[q, q] = c; G[p, q] = s; G[q, p] = -s
    cpu = A.astype(np.float64) @ G

    err = float(np.max(np.abs(gpu.astype(np.float64) - cpu)))
    check("apply_col_rotation matches Givens", err < 1e-4, f"max abs err {err:.2e}")
    # orthogonal rotation must preserve Frobenius norm
    dn = abs(np.linalg.norm(gpu) - np.linalg.norm(A))
    check("rotation preserves ‖·‖_F", dn < 1e-2, f"Δ‖·‖ {dn:.2e}")


def test_harness_metrics():
    print("3) Accuracy harness self-check (exact LAPACK should score ~0)")
    rng = np.random.default_rng(2)
    A = rng.standard_normal((40, 40)).astype("f4"); A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    check("recon_error_eigh ~ 0", ref.recon_error_eigh(A, w, V) < 1e-5,
          f"{ref.recon_error_eigh(A, w, V):.2e}")
    check("orthogonality_error ~ 0", ref.orthogonality_error(V) < 1e-4,
          f"{ref.orthogonality_error(V):.2e}")

    B = rng.standard_normal((50, 30)).astype("f4")
    U, S, Vh = np.linalg.svd(B, full_matrices=False)
    check("recon_error_svd ~ 0", ref.recon_error_svd(B, U, S, Vh) < 1e-5,
          f"{ref.recon_error_svd(B, U, S, Vh):.2e}")
    check("values_rel_error self ~ 0", ref.values_rel_error(S, S) < 1e-6)


def test_case_generators():
    print("4) Pathological test-matrix generators")
    sym = ref.symmetric_cases(n=64)
    check("symmetric_cases complete", len(sym) == 8, f"{sorted(sym)}")
    check("symmetric cases are symmetric",
          all(np.allclose(M, M.T, atol=1e-3) for M in sym.values()))
    svd = ref.svd_cases(m=80, n=48)
    check("svd_cases complete", len(svd) == 5, f"{sorted(svd)}")


def test_placeholders_run():
    print("5) eigh/svd placeholders run through the harness (CPU fallback)")
    A = ref.symmetric_cases(n=64)["random_sym"]
    w, V = metal_eigh(torch.from_numpy(A))
    e = ref.recon_error_eigh(A, w.cpu().numpy(), V.cpu().numpy())
    check("metal_eigh placeholder reconstructs", e < 1e-4, f"recon {e:.2e}")
    B = ref.svd_cases()["tall"]
    U, S, Vh = metal_svd(torch.from_numpy(B))
    e = ref.recon_error_svd(B, U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy())
    check("metal_svd placeholder reconstructs", e < 1e-4, f"recon {e:.2e}")


def main():
    print(f"torch {torch.__version__} | mps={torch.backends.mps.is_available()}\n")
    test_integration()
    test_rotation_primitive()
    test_harness_metrics()
    test_case_generators()
    test_placeholders_run()
    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
