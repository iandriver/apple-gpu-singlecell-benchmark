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

from ._dispatch import _as_mps_f32, get_lib, get_lib_variant, grid_1d

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


# ── batched eigh (Phase 2: the actual GPU-speedup path) ────────────────────
BATCH_N_MAX = 64       # largest n the batched path supports (32< n <=64 via V-global)


def _bucket(n: int) -> int:
    """Smallest threadgroup footprint that holds n (fully-resident path, n<=32)."""
    for b in (8, 16, 32):
        if n <= b:
            return b
    raise AssertionError(f"fully-resident path needs n <= 32 (got {n})")


def _default_btg(n: int) -> int:
    """Threads per matrix, autotuned for occupancy (see tune_batched.py).

    Fewer threads/matrix -> more matrices resident per core -> higher throughput,
    until there are too few threads to cover the work. Measured optima on M5 Pro.
    """
    return {8: 16, 16: 32, 32: 32}[_bucket(n)]


def batched_eigh(A: torch.Tensor, max_sweeps: int = 30, tol: float = 1e-6,
                 max_bn: int | None = None, btg: int | None = None,
                 store: str = "fp32"):
    """Batched symmetric eigh for many small matrices, one GPU threadgroup each.

    A : (B, n, n), n <= 64.  Returns (w (B,n) ascending, V (B,n,n)).
    n <= 32 uses the fully threadgroup-resident kernel (A and V on-chip); 32 < n <= 64
    keeps A on-chip and V in device memory. `store="fp16"` halves the on-chip storage
    (compute stays fp32) for higher occupancy, at reduced accuracy. max_bn / btg
    override the autotuned config.
    """
    assert A.ndim == 3 and A.shape[1] == A.shape[2], "(B, n, n) expected"
    B, n, _ = A.shape
    assert n <= BATCH_N_MAX, f"batched path supports n <= {BATCH_N_MAX} (got {n})"
    assert store in ("fp32", "fp16")
    out_dev = A.device if A.device.type == "mps" else "cpu"

    Aw = A.to(device="mps", dtype=torch.float32).contiguous().clone()
    V = torch.empty_like(Aw)

    if n <= 32:
        defines = {"BATCH_MAX_BN": max_bn or _bucket(n), "BATCH_BTG": btg or _default_btg(n)}
        if store == "fp16":
            defines["BATCH_STORE"] = "half"
        g = defines["BATCH_BTG"]
        lib = get_lib_variant(defines)
        lib.batched_jacobi_eigh(Aw, V, int(n), int(max_sweeps), float(tol),
                                threads=(g * B,), group_size=(g,))
    else:
        g = btg or 64
        lib = get_lib_variant({"BATCH_BIG_MAXN": max_bn or (48 if n <= 48 else 64),
                               "BATCH_BIG_BTG": g})
        lib.batched_jacobi_eigh_vg(Aw, V, int(n), int(max_sweeps), float(tol),
                                   threads=(g * B,), group_size=(g,))

    w = torch.diagonal(Aw, dim1=-2, dim2=-1)           # (B, n)
    order = torch.argsort(w, dim=-1)
    w = torch.gather(w, -1, order)
    V = torch.gather(V, 2, order.unsqueeze(1).expand(B, n, n))
    return w.to(out_dev), V.to(out_dev)


# ── batched SVD (Phase 3: one-sided Jacobi, the GPU-speedup path for SVD) ──
SVD_M_MAX = 64         # must match `SVD_MAXM` in metal
SVD_N_MAX = 32         # must match `SVD_MAXN`
SVD_BTG = 32           # must match `SVD_BTG`


def _run_svd_tall(A: torch.Tensor, max_sweeps: int, tol: float):
    """One-sided Jacobi SVD for tall batches (m >= n). Returns U,S,V sorted desc."""
    B, m, n = A.shape
    assert m >= n and m <= SVD_M_MAX and n <= SVD_N_MAX, \
        f"tall path needs n<=m, m<={SVD_M_MAX}, n<={SVD_N_MAX} (got {m}x{n})"
    lib = get_lib()
    U = A.to(device="mps", dtype=torch.float32).contiguous().clone()  # -> U
    V = torch.empty(B, n, n, device="mps", dtype=torch.float32)
    S = torch.empty(B, n, device="mps", dtype=torch.float32)
    lib.batched_jacobi_svd(U, V, S, int(m), int(n), int(max_sweeps), float(tol),
                           threads=(SVD_BTG * B,), group_size=(SVD_BTG,))
    order = torch.argsort(S, dim=-1, descending=True)
    S = torch.gather(S, 1, order)
    U = torch.gather(U, 2, order.unsqueeze(1).expand(B, m, n))
    V = torch.gather(V, 2, order.unsqueeze(1).expand(B, n, n))
    return U, S, V


def batched_svd(A: torch.Tensor, max_sweeps: int = 30, tol: float = 1e-6):
    """Batched reduced SVD for many small matrices, one GPU threadgroup each.

    A : (B, m, n).  Returns (U (B,m,k), S (B,k), Vh (B,k,n)), k=min(m,n), matching
    torch.linalg.svd(A, full_matrices=False). Tall handled directly; wide via
    transpose. Supports max(m,n) <= 64 and min(m,n) <= 32.
    """
    assert A.ndim == 3, "(B, m, n) expected"
    B, m, n = A.shape
    out_dev = A.device if A.device.type == "mps" else "cpu"

    if m >= n:
        U, S, V = _run_svd_tall(A, max_sweeps, tol)
        Vh = V.transpose(-2, -1)
    else:
        # A = (Aᵀ)ᵀ; tall-SVD of Aᵀ gives (Ut,S,Vt) -> U_A = Vt, Vh_A = Utᵀ
        Ut, S, Vt = _run_svd_tall(A.transpose(-2, -1).contiguous(), max_sweeps, tol)
        U, Vh = Vt, Ut.transpose(-2, -1)
    return U.to(out_dev), S.to(out_dev), Vh.to(out_dev)


def metal_svd(A: torch.Tensor, full_matrices: bool = False):
    """SVD. PLACEHOLDER: CPU fallback until Phase 1."""
    _warn_once("svd", "metal_svd is a Phase-0 placeholder (CPU LAPACK). "
                      "Phase 1 swaps in the one-sided Jacobi Metal kernel.")
    U, S, Vh = torch.linalg.svd(A.detach().to("cpu", torch.float32),
                                full_matrices=full_matrices)
    dev = A.device if A.device.type == "mps" else "cpu"
    return U.to(dev), S.to(dev), Vh.to(dev)
