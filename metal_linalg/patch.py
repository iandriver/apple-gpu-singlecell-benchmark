"""
Phase 5/7: a complete torch.linalg-on-MPS shim.

PyTorch's MPS backend covers most ops (matmul, conv, attention, elementwise, and
even inv/solve/cholesky/solve_triangular), but the matrix FACTORIZATIONS are missing
or broken on MPS: eigh raises, qr hangs, svd silently falls back to CPU, etc. That
gap makes a lot of scientific/ML code fail or silently leave the GPU.

`install()` patches the torch.linalg factorization/solver surface so it works on MPS
tensors, leaving CPU tensors untouched:

  * eigh / eigvalsh / svd / svdvals : GPU-accelerated for batched-small matrices via
    metal_linalg's Jacobi kernels; CPU round-trip otherwise.
  * qr / eig / eigvals / lstsq / pinv / matrix_rank / slogdet : transparent CPU
    round-trip (we have no GPU kernel; this just makes the call succeed on MPS).

The round-trip moves inputs to CPU, runs the original LAPACK routine, and moves the
result back to the input device and dtype — preserving the structseq return type so
`.eigenvalues`, unpacking, etc. all still work. We FORCE the CPU path for unsupported
ops rather than "try MPS first", because some (qr) hang instead of raising.
"""

from __future__ import annotations

import contextlib

import torch

from .accel import gpu_matrix_rank, gpu_pinv
from .dispatch import eigh as _fast_eigh
from .kernels import (BATCH_N_MAX, SVD_M_MAX, SVD_N_MAX, batched_eigh,
                      batched_svd)

_orig = {}   # name -> original torch.linalg callable

# torch.linalg functions with no GPU win here -> CPU round-trip on MPS. We accelerate
# an op only when its CPU baseline is SVD-bound (pinv, matrix_rank); ops whose CPU path
# is cheap QR (qr itself, lstsq) are FASTER on CPU, so they round-trip. (Measured in
# test_phase8: GPU CholeskyQR ~0.08x, GPU-SVD lstsq ~0.73x — both losers.)
_FALLBACK_OPS = ["qr", "lstsq", "eig", "eigvals", "slogdet", "matrix_power", "cond"]


# ── helpers ─────────────────────────────────────────────────────────────────
def _ret(name, values):
    """Rebuild a torch.return_types structseq so attribute access still works."""
    rt = getattr(torch.return_types, name, None)
    try:
        return rt(values) if rt is not None else tuple(values)
    except TypeError:
        return tuple(values)


def _cpu_dtype(dt):
    # LAPACK has no fp16/bf16 -> upcast for the CPU computation, restore on return.
    return torch.float32 if dt in (torch.float16, torch.bfloat16) else dt


def _roundtrip(fn, args, kwargs):
    """Run fn on CPU for MPS inputs, move tensor outputs back to the input device."""
    dev, dt = None, None
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor) and a.device.type == "mps":
            dev, dt = a.device, a.dtype
            break

    def down(x):
        return x.detach().to("cpu", _cpu_dtype(x.dtype)) if isinstance(x, torch.Tensor) else x

    out = fn(*[down(a) for a in args], **{k: down(v) for k, v in kwargs.items()})

    def up(x):
        if not isinstance(x, torch.Tensor):
            return x
        y = x.to(dev)
        if dt in (torch.float16, torch.bfloat16) and y.is_floating_point():
            y = y.to(dt)
        return y

    if isinstance(out, torch.Tensor):
        return up(out)
    moved = [up(x) for x in out]
    try:
        return type(out)(moved)
    except TypeError:
        return tuple(moved)


def _is_mps(x):
    return isinstance(x, torch.Tensor) and x.device.type == "mps"


def _eigh_in_range(A):
    return A.ndim == 3 and A.shape[-1] == A.shape[-2] and A.shape[-1] <= BATCH_N_MAX


def _svd_in_range(A, full_matrices):
    m, n = A.shape[-2], A.shape[-1]
    return (A.ndim == 3 and max(m, n) <= SVD_M_MAX and min(m, n) <= SVD_N_MAX
            and (not full_matrices or m == n))


# ── accelerated wrappers ────────────────────────────────────────────────────
def _eigh(A, UPLO="L"):
    if _is_mps(A):
        if _eigh_in_range(A):
            return _ret("linalg_eigh", _fast_eigh(0.5 * (A + A.transpose(-2, -1))))
        return _roundtrip(_orig["eigh"], (A,), {"UPLO": UPLO})
    return _orig["eigh"](A, UPLO=UPLO)


def _eigvalsh(A, UPLO="L"):
    if _is_mps(A):
        if _eigh_in_range(A):
            return _fast_eigh(0.5 * (A + A.transpose(-2, -1)))[0]
        return _roundtrip(_orig["eigvalsh"], (A,), {"UPLO": UPLO})
    return _orig["eigvalsh"](A, UPLO=UPLO)


def _svd(A, full_matrices=True, *, driver=None):
    if _is_mps(A):
        if _svd_in_range(A, full_matrices):
            return _ret("linalg_svd", batched_svd(A))
        return _roundtrip(_orig["svd"], (A,), {"full_matrices": full_matrices})
    return _orig["svd"](A, full_matrices=full_matrices, driver=driver)


def _svdvals(A, *, driver=None):
    if _is_mps(A):
        if _svd_in_range(A, full_matrices=False):
            return batched_svd(A)[1]
        return _roundtrip(_orig["svdvals"], (A,), {})
    return _orig["svdvals"](A, driver=driver)


def _pinv(A, *args, **kwargs):
    # accelerate only the plain pinv(A) batched-small call; anything fancier -> CPU
    if _is_mps(A) and not args and not kwargs and _svd_in_range(A, full_matrices=False):
        return gpu_pinv(A)
    if _is_mps(A):
        return _roundtrip(_orig["pinv"], (A, *args), kwargs)
    return _orig["pinv"](A, *args, **kwargs)


def _matrix_rank(A, *args, **kwargs):
    if _is_mps(A) and not args and not kwargs and _svd_in_range(A, full_matrices=False):
        return gpu_matrix_rank(A)
    if _is_mps(A):
        return _roundtrip(_orig["matrix_rank"], (A, *args), kwargs)
    return _orig["matrix_rank"](A, *args, **kwargs)


def _make_fallback(name):
    def wrapper(*args, **kwargs):
        if args and _is_mps(args[0]):
            return _roundtrip(_orig[name], args, kwargs)
        return _orig[name](*args, **kwargs)
    wrapper.__name__ = name
    return wrapper


# ── install / uninstall ───────────────────────────────────────────────────--
_ACCEL = {"eigh": _eigh, "eigvalsh": _eigvalsh, "svd": _svd, "svdvals": _svdvals,
          "pinv": _pinv, "matrix_rank": _matrix_rank}


def install():
    """Patch the torch.linalg factorization surface to work on MPS tensors."""
    if _orig:
        return
    for name in list(_ACCEL) + _FALLBACK_OPS:
        fn = getattr(torch.linalg, name, None)
        if fn is None:
            continue
        _orig[name] = fn
        setattr(torch.linalg, name, _ACCEL.get(name) or _make_fallback(name))


def uninstall():
    """Restore all original torch.linalg functions."""
    for name, fn in _orig.items():
        setattr(torch.linalg, name, fn)
    _orig.clear()


@contextlib.contextmanager
def patched():
    install()
    try:
        yield
    finally:
        uninstall()
