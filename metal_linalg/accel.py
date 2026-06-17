"""
GPU-accelerated linalg ops built from primitives MPS *does* support — matmul,
cholesky, solve_triangular — plus our batched Jacobi SVD. These replace the CPU
round-trip for qr / pinv / lstsq / matrix_rank with real on-GPU computation.

All functions run entirely on MPS and accept single (m,n) or batched (B,m,n) inputs.
qr uses CholeskyQR2 (Cholesky-QR run twice for orthogonality, since Cholesky-QR
squares the condition number); it raises if the matrix is too ill-conditioned /
rank-deficient for Cholesky, so the caller can fall back to CPU.
"""

from __future__ import annotations

import torch

from ._dispatch import get_lib_variant
from .kernels import batched_svd

QR_M_MAX = 64      # matches QR_MAXM in kernels.metal
QR_N_MAX = 32      # matches QR_MAXN


def householder_qr(A: torch.Tensor, btg: int = 64):
    """Batched reduced QR via the Metal Householder kernel (one threadgroup/matrix).

    A: (B,m,n) tall, m>=n, m<=64, n<=32. Returns (Q (B,m,n), R (B,n,n)) on MPS.
    Backward-stable Householder (unlike CholeskyQR); single direct pass.
    """
    assert A.ndim == 3 and A.shape[-2] >= A.shape[-1]
    B, m, n = A.shape
    assert m <= QR_M_MAX and n <= QR_N_MAX
    lib = get_lib_variant({"QR_MAXM": QR_M_MAX, "QR_MAXN": QR_N_MAX, "QR_BTG": btg})
    Aw = A.to(device="mps", dtype=torch.float32).contiguous()
    Q = torch.empty(B, m, n, device="mps", dtype=torch.float32)
    R = torch.empty(B, n, n, device="mps", dtype=torch.float32)
    lib.batched_householder_qr(Aw, Q, R, int(m), int(n),
                               threads=(btg * B,), group_size=(btg,))
    return Q, R


def _batched(A):
    return (A.unsqueeze(0), True) if A.ndim == 2 else (A, False)


def gpu_qr(A: torch.Tensor):
    """Reduced QR (A = Q R, Q orthonormal columns, R upper) on the GPU via CholeskyQR2.

    A: (m,n) or (B,m,n) with m >= n. Returns (Q, R) on MPS. Raises if Cholesky fails
    (rank-deficient / very ill-conditioned) so the caller can fall back.

    NOTE — measured SLOWER than CPU for batched-tiny (~0.08x in test_phase8): it relies
    on torch's native MPS cholesky/solve_triangular, which are slow over many small
    matrices. The patch therefore keeps qr on the CPU round-trip. Kept here as a correct,
    available implementation (and useful for a single larger matrix).
    """
    X, squeezed = _batched(A.to(device="mps", dtype=torch.float32))

    def cqr(M):
        G = M.transpose(-2, -1) @ M                       # (B,n,n) SPD
        R = torch.linalg.cholesky(G).transpose(-2, -1)    # upper, RᵀR = G
        # Q = M R⁻¹  via  Rᵀ Qᵀ = Mᵀ  (Rᵀ lower-triangular)
        Qt = torch.linalg.solve_triangular(R.transpose(-2, -1), M.transpose(-2, -1),
                                           upper=False)
        return Qt.transpose(-2, -1), R

    Q1, R1 = cqr(X)
    Q2, R2 = cqr(Q1)                                      # reorthogonalize
    Q, R = Q2, R2 @ R1
    return (Q.squeeze(0), R.squeeze(0)) if squeezed else (Q, R)


def gpu_pinv(A: torch.Tensor, rcond: float | None = None):
    """Moore-Penrose pseudoinverse via our GPU batched SVD. A:(B,m,n)->(B,n,m)."""
    X, squeezed = _batched(A)
    U, S, Vh = batched_svd(X)                             # reduced SVD on GPU
    m, n = X.shape[-2], X.shape[-1]
    if rcond is None:
        rcond = torch.finfo(torch.float32).eps * max(m, n)
    tol = rcond * S.amax(dim=-1, keepdim=True)
    Sinv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
    P = (Vh.transpose(-2, -1) * Sinv.unsqueeze(-2)) @ U.transpose(-2, -1)
    return P.squeeze(0) if squeezed else P


def gpu_lstsq(A: torch.Tensor, B: torch.Tensor):
    """Min-norm least-squares solution via our GPU batched SVD: x = V diag(1/s) Uᵀ b.

    Routed through the Metal SVD kernel (not CholeskyQR): on MPS, torch's native
    batched cholesky/solve_triangular over many tiny matrices is slow, so a QR-based
    path loses to the CPU — the SVD path wins because it uses our custom kernel.
    """
    Xa, sq = _batched(A.to(device="mps", dtype=torch.float32))
    Xb, _ = _batched(B.to(device="mps", dtype=torch.float32))
    U, S, Vh = batched_svd(Xa)                           # reduced SVD on GPU
    m, n = Xa.shape[-2], Xa.shape[-1]
    tol = torch.finfo(torch.float32).eps * max(m, n) * S.amax(dim=-1, keepdim=True)
    Sinv = torch.where(S > tol, 1.0 / S, torch.zeros_like(S))
    x = Vh.transpose(-2, -1) @ (Sinv.unsqueeze(-1) * (U.transpose(-2, -1) @ Xb))
    return x.squeeze(0) if sq else x


def gpu_matrix_rank(A: torch.Tensor, tol: float | None = None):
    """Numerical rank from GPU singular values."""
    X, squeezed = _batched(A)
    S = batched_svd(X)[1]
    m, n = X.shape[-2], X.shape[-1]
    if tol is None:
        tol = torch.finfo(torch.float32).eps * max(m, n)
    thresh = tol * S.amax(dim=-1, keepdim=True)
    rank = (S > thresh).sum(dim=-1)
    return rank.squeeze(0) if squeezed else rank
