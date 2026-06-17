"""
Phase 5: drop-in integration. `install()` monkey-patches torch.linalg.eigh and
torch.linalg.svd so existing code transparently gets the GPU kernels on MPS tensors.

Design: we ONLY intercept MPS tensors — where torch.linalg currently raises
"not implemented for MPS" anyway. CPU tensors pass straight through to the original
LAPACK path, so nothing about existing CPU behavior changes. On MPS:
  * batched small matrices -> our fast Jacobi kernels (the speedup)
  * anything else -> compute on CPU and move the result back to the input device
    (so the call succeeds instead of raising)

Usage:
    import metal_linalg
    metal_linalg.install()                 # patch globally
    w, V = torch.linalg.eigh(A_on_mps)     # now works + fast for batched-small
    metal_linalg.uninstall()
  or:
    with metal_linalg.patched():
        ...
"""

from __future__ import annotations

import contextlib

import torch

from .dispatch import eigh as _fast_eigh
from .kernels import SVD_M_MAX, SVD_N_MAX, batched_svd

_orig = {}


def _ret(kind, values):
    """Best-effort torch.return_types structseq so `.eigenvalues` etc. still work."""
    rt = getattr(torch.return_types, kind, None)
    try:
        return rt(values) if rt is not None else tuple(values)
    except TypeError:
        return tuple(values)


def _patched_eigh(A, UPLO="L"):
    if isinstance(A, torch.Tensor) and A.device.type == "mps":
        As = 0.5 * (A + A.transpose(-2, -1))          # honor eigh's symmetric contract
        w, V = _fast_eigh(As)
        return _ret("eigh", (w, V))
    return _orig["eigh"](A, UPLO=UPLO)


def _patched_svd(A, full_matrices=True, *, driver=None):
    if isinstance(A, torch.Tensor) and A.device.type == "mps":
        m, n = A.shape[-2], A.shape[-1]
        in_range = (A.ndim == 3 and max(m, n) <= SVD_M_MAX and min(m, n) <= SVD_N_MAX)
        if in_range and (not full_matrices or m == n):   # reduced == full when square
            U, S, Vh = batched_svd(A)
            return _ret("svd", (U, S, Vh))
        U, S, Vh = _orig["svd"](A.detach().to("cpu", torch.float32),
                                full_matrices=full_matrices)
        return _ret("svd", (U.to(A.device), S.to(A.device), Vh.to(A.device)))
    return _orig["svd"](A, full_matrices=full_matrices, driver=driver)


def install():
    """Patch torch.linalg.eigh / svd to use the Metal kernels on MPS tensors."""
    if _orig:
        return
    _orig["eigh"] = torch.linalg.eigh
    _orig["svd"] = torch.linalg.svd
    torch.linalg.eigh = _patched_eigh
    torch.linalg.svd = _patched_svd


def uninstall():
    """Restore the original torch.linalg functions."""
    if not _orig:
        return
    torch.linalg.eigh = _orig.pop("eigh")
    torch.linalg.svd = _orig.pop("svd")


@contextlib.contextmanager
def patched():
    install()
    try:
        yield
    finally:
        uninstall()
