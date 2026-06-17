"""
Integration scaffold: compile the Metal kernels once and dispatch them on MPS
tensors, zero-copy. This mirrors the torch.mps.compile_shader() pattern proven by
github.com/tashiscool/fp8-mps-metal — runtime shader compilation, no Xcode.

Everything else in metal_linalg builds on get_lib(); Phase 1's eigensolver/SVD
kernels get appended to kernels.metal and dispatched through the same path.
"""

from __future__ import annotations

import os

import torch

_LIB = None
_SOURCE = None


def mps_available() -> bool:
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def _load_source() -> str:
    global _SOURCE
    if _SOURCE is None:
        path = os.path.join(os.path.dirname(__file__), "kernels.metal")
        with open(path) as f:
            _SOURCE = f.read()
    return _SOURCE


def get_lib():
    """Compiled Metal library (singleton). Raises if MPS/compile_shader missing."""
    global _LIB
    if _LIB is None:
        if not mps_available():
            raise RuntimeError(
                "Requires Apple Silicon + macOS with PyTorch MPS and "
                "torch.mps.compile_shader (PyTorch >= 2.10)."
            )
        _LIB = torch.mps.compile_shader(_load_source())
    return _LIB


def _as_mps_f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(device="mps", dtype=torch.float32).contiguous()


def grid_1d(n: int, group: int = 256):
    """(threads, group_size) for a simple 1-D launch covering n elements."""
    return (int(n),), (int(min(group, n)),)
