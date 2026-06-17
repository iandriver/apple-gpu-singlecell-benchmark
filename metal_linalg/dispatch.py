"""
Phase 4: robust auto-dispatch. `eigh` / `svd` pick the fast GPU batched path when
it applies (batched input within the supported size range) and fall back to CPU
LAPACK otherwise — so callers get the speedup automatically and a correct answer
always.

Robustness:
  * size/shape routing: only batched small matrices go to the GPU kernel; single
    matrices and out-of-range sizes use CPU (where they're faster / supported).
  * non-convergence guard (verify=True): recompute the max reconstruction residual
    on the GPU result and, if it exceeds a tolerance, recompute *those* matrices on
    the CPU. Cheap insurance against a pathological matrix that didn't converge in
    max_sweeps. Off by default (keeps the hot path lean); on when correctness must
    be guaranteed.
"""

from __future__ import annotations

import torch

from .kernels import (BATCH_N_MAX, SVD_M_MAX, SVD_N_MAX, batched_eigh,
                      batched_svd)


def _recon_resid_eigh(A, w, V):
    """Per-matrix ‖A − V diag(w) Vᵀ‖_F / ‖A‖_F  (batched, on whatever device)."""
    Ar = V @ (w.unsqueeze(-1) * V.transpose(-2, -1))
    num = torch.linalg.norm(A - Ar, dim=(-2, -1))
    den = torch.linalg.norm(A, dim=(-2, -1)).clamp_min(1e-30)
    return num / den


def eigh(A: torch.Tensor, store: str = "fp32", verify: bool = False,
         resid_tol: float = 1e-3):
    """Symmetric eigendecomposition with automatic GPU/CPU routing.

    Batched (B,n,n) with n<=64  -> GPU Jacobi (the fast path).
    Otherwise (single matrix, n>64) -> CPU LAPACK.
    Returns (eigenvalues ascending, eigenvectors), matching torch.linalg.eigh.
    """
    if A.ndim == 3 and A.shape[-1] == A.shape[-2] and A.shape[-1] <= BATCH_N_MAX:
        w, V = batched_eigh(A, store=store)
        if verify:
            bad = _recon_resid_eigh(A.to(w.device, w.dtype), w, V) > resid_tol
            if bool(bad.any()):
                idx = bad.nonzero(as_tuple=True)[0]
                wc, Vc = torch.linalg.eigh(A[idx].to("cpu", torch.float32))
                w[idx], V[idx] = wc.to(w.device), Vc.to(V.device)
        return w, V
    return torch.linalg.eigh(A.to("cpu", torch.float32))


def svd(A: torch.Tensor):
    """Reduced SVD with automatic GPU/CPU routing.

    Batched (B,m,n) with max(m,n)<=64 and min(m,n)<=32 -> GPU one-sided Jacobi.
    Otherwise -> CPU LAPACK. Returns (U, S, Vh) like torch.linalg.svd(full_matrices=False).
    """
    if A.ndim == 3:
        m, n = A.shape[-2], A.shape[-1]
        if max(m, n) <= SVD_M_MAX and min(m, n) <= SVD_N_MAX:
            return batched_svd(A)
    return torch.linalg.svd(A.to("cpu", torch.float32), full_matrices=False)
