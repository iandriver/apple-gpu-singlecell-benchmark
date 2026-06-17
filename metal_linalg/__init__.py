"""GPU linear algebra on Apple Metal — eigh/svd via Jacobi kernels.

Phase 0 (current): integration scaffold + accuracy/benchmark harness. The
saxpy / apply_col_rotation kernels prove the torch.mps.compile_shader path;
metal_eigh / metal_svd are CPU-fallback placeholders until Phase 1.
"""

from ._dispatch import get_lib, mps_available
from .kernels import apply_col_rotation, metal_eigh, metal_svd, saxpy

__all__ = ["mps_available", "get_lib", "saxpy", "apply_col_rotation",
           "metal_eigh", "metal_svd"]
