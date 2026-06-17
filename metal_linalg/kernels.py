"""
Python wrappers for the Metal kernels, plus the eigh/svd entry points.

Phase 0 status:
  * saxpy, apply_col_rotation  -> real Metal kernels (prove the integration path)
  * metal_eigh, metal_svd      -> PLACEHOLDERS that fall back to CPU LAPACK.
    Phase 1 replaces their internals with the in-threadgroup Jacobi kernels; the
    signatures, dispatch wiring, and the accuracy/benchmark harness around them
    are exercised now so nothing about the integration is left to discover later.
"""

from __future__ import annotations

import torch

from ._dispatch import _as_mps_f32, get_lib, grid_1d

_warned = set()


def _warn_once(key: str, msg: str):
    if key not in _warned:
        _warned.add(key)
        print(f"[metal_linalg] {msg}")


# ── Real Metal kernels (Phase 0) ───────────────────────────────────────────
def saxpy(x: torch.Tensor, y: torch.Tensor, a: float) -> torch.Tensor:
    """out = a*x + y, computed by a Metal kernel on the GPU."""
    lib = get_lib()
    x = _as_mps_f32(x).view(-1)
    y = _as_mps_f32(y).view(-1)
    out = torch.empty_like(x)
    n = x.numel()
    threads, group = grid_1d(n)
    lib.saxpy(x, y, out, float(a), n, threads=threads, group_size=group)
    return out.view_as(y)


def apply_col_rotation(A: torch.Tensor, p: int, q: int, c: float, s: float) -> torch.Tensor:
    """In-place Givens rotation of columns p,q of square A (Metal kernel)."""
    lib = get_lib()
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "square matrix expected"
    A = _as_mps_f32(A)
    n = A.shape[0]
    threads, group = grid_1d(n)
    lib.apply_col_rotation(A, int(n), int(p), int(q), float(c), float(s),
                           threads=threads, group_size=group)
    return A


# ── eigh entry point (Phase 1: GPU Jacobi for n <= EIGH_GPU_MAX) ───────────
# Single-threadgroup Jacobi; correctness-first. Above the cap we fall back to CPU
# (Phase 2 adds the multi-threadgroup path for large n).
EIGH_TG = 256          # threads per threadgroup (must match `constant TG` in metal)
EIGH_GPU_MAX = 256     # Phase 1 scope


def metal_eigh(A: torch.Tensor, max_sweeps: int = 30, tol: float = 1e-6,
               force_gpu: bool = False):
    """Symmetric eigendecomposition -> (eigenvalues ascending, eigenvectors).

    GPU two-sided Jacobi for n <= EIGH_GPU_MAX; CPU LAPACK fallback above.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "square matrix expected"
    n = A.shape[0]
    out_dev = A.device if A.device.type == "mps" else "cpu"

    if n > EIGH_GPU_MAX and not force_gpu:
        _warn_once("eigh_cpu", f"metal_eigh: n={n} > {EIGH_GPU_MAX}, using CPU "
                               f"fallback (Phase 2 covers large n).")
        w, V = torch.linalg.eigh(A.detach().to("cpu", torch.float32))
        return w.to(out_dev), V.to(out_dev)

    lib = get_lib()
    Aw = _as_mps_f32(A).clone()                       # destroyed -> diagonal = eigvals
    V = torch.eye(n, device="mps", dtype=torch.float32)
    lib.jacobi_eigh(Aw, V, int(n), int(max_sweeps), float(tol),
                    threads=(EIGH_TG,), group_size=(EIGH_TG,))
    w = torch.diagonal(Aw).clone()
    order = torch.argsort(w)                           # ascending, LAPACK convention
    return w[order].to(out_dev), V[:, order].to(out_dev)


def metal_svd(A: torch.Tensor, full_matrices: bool = False):
    """SVD. PLACEHOLDER: CPU fallback until Phase 1."""
    _warn_once("svd", "metal_svd is a Phase-0 placeholder (CPU LAPACK). "
                      "Phase 1 swaps in the one-sided Jacobi Metal kernel.")
    U, S, Vh = torch.linalg.svd(A.detach().to("cpu", torch.float32),
                                full_matrices=full_matrices)
    dev = A.device if A.device.type == "mps" else "cpu"
    return U.to(dev), S.to(dev), Vh.to(dev)
