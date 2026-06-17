# metal_linalg — general-purpose GPU `eigh` / `svd` on Apple Metal

PyTorch-MPS and MLX both lack a GPU eigendecomposition / SVD (see the parent
[benchmark](../README.md)). This subproject builds one with custom Metal kernels,
dispatched via `torch.mps.compile_shader()` (runtime compile, zero-copy on MPS
tensors, no Xcode — the mechanism proven by
[fp8-mps-metal](https://github.com/tashiscool/fp8-mps-metal)).

**Algorithm:** Jacobi methods — two-sided (cyclic) Jacobi for `eigh`, one-sided
(Hestenes) Jacobi for `svd`. Rationale: they produce singular/eigen **vectors**
naturally, handle rectangular matrices, are accurate for small singular values,
and are far simpler to implement correctly in shader code than the
Householder→bidiagonal→divide-and-conquer pipeline. (Even the SOTA research effort
on Metal SVD, [NextLA.jl](https://arxiv.org/html/2508.06339v1), produced singular
*values only*, square only — which is exactly what Jacobi avoids.)

## Status

| Phase | Scope | State |
|---|---|---|
| **0** | **Integration scaffold + accuracy/benchmark harness** | ✅ **done** |
| **1** | **eigh — two-sided Jacobi, single threadgroup (the go/no-go)** | ✅ **done — GO** |
| 2 | eigh at scale — multi-threadgroup / tiling for speed + large n | next |
| 3 | svd — one-sided Jacobi (+ QR precondition for tall) | planned |
| 4 | robustness & precision (fp16, scaling, CPU fallback) | planned |
| 5 | integration: patch `torch.linalg.{svd,eigh}` on MPS, batching, packaging | planned |

## What Phase 0 delivered

- **Proven integration path** — `torch.mps.compile_shader` compiles our Metal
  source on this machine (torch 2.12, M5 Pro); a trivial `saxpy` kernel is
  bit-exact and the forward-looking **`apply_col_rotation`** (a Givens/Jacobi
  rotation, the Phase 1 building block) matches NumPy to fp32 epsilon — zero-copy,
  in-place on an MPS tensor.
- **Accuracy harness** ([`reference.py`](reference.py)) — reconstruction,
  orthogonality, and value-relative-error metrics, plus pathological test
  matrices (clustered/degenerate eigenvalues, rank-deficient, ill-conditioned,
  extreme scale, tall/wide/square). Phase 1 kernels are graded against these.
- **Benchmark harness** ([`bench.py`](bench.py)) — CPU/Accelerate baselines the
  Metal kernels must beat (e.g. `eigh` 1024×1024 ≈ 42 ms on this machine).
- **Entry points** — `metal_eigh` / `metal_svd` exist with final signatures but
  are CPU-fallback **placeholders**; Phase 1 swaps in the kernels with no harness
  changes.

## What Phase 1 delivered (GO)

A working GPU symmetric eigensolver: `jacobi_eigh` in [`kernels.metal`](kernels.metal),
a single threadgroup running two-sided cyclic Jacobi (column+row rotations, eigenvector
accumulation, off-diagonal-norm convergence). Wired into `metal_eigh` for n ≤ 256 (CPU
fallback above, until Phase 2).

**Correctness vs LAPACK** ([`test_phase1.py`](test_phase1.py), 14/14):

| case (n=64) | reconstruction | orthogonality | eigenvalues |
|---|--:|--:|--:|
| random symmetric | 2.0e-6 | 9.5e-6 | 1.7e-6 |
| clustered eigenvalues | 7.2e-6 | 1.9e-5 | 1.5e-5 |
| ill-conditioned (1e0–1e8) | 1.3e-6 | 1.3e-5 | 1.2e-6 |
| rank-deficient | 1.6e-6 | 8.8e-6 | 7.6e-7 |
| tiny / huge scale (1e±12) | ~2e-6 | ~9e-6 | ~2e-6 |

Holds across n = 8…256 — textbook fp32-Jacobi accuracy, hard cases included.

**Speed is not the bar yet** (and a single threadgroup uses a sliver of the GPU):
GPU 217 ms vs CPU/AMX 2.3 ms at n=256. Small matrices favor the CPU; the GPU win
comes in Phase 2 (multi-threadgroup, large n) and from batching many small matrices
(one threadgroup each) in Phase 5. The go/no-go was **correctness — passed**.

## Files

| File | What |
|---|---|
| [`kernels.metal`](kernels.metal) | Metal source. Phase 0: `saxpy`, `apply_col_rotation`. Phase 1: `jacobi_eigh`. |
| [`_dispatch.py`](_dispatch.py) | `compile_shader` singleton + MPS-tensor dispatch helpers. |
| [`kernels.py`](kernels.py) | Python wrappers + `metal_eigh`/`metal_svd` entry points. |
| [`reference.py`](reference.py) | Accuracy metrics + pathological test matrices. |
| [`bench.py`](bench.py) | CPU-baseline benchmark harness. |
| [`test_phase0.py`](test_phase0.py) | Phase 0 acceptance test (13 checks). |
| [`test_phase1.py`](test_phase1.py) | Phase 1 acceptance test: GPU Jacobi eigh vs LAPACK (14 checks). |

## Run

```bash
# from the repo root, using the project venv
python -m metal_linalg.test_phase0    # integration scaffold  -> 13 passed
python -m metal_linalg.test_phase1    # GPU Jacobi eigh vs LAPACK -> 14 passed
python -m metal_linalg.bench          # CPU baselines
```

## Scope notes

- **fp32 primary** (fp16 opportunistic). Apple GPUs have no native fp64 — that's
  the precision ceiling unless double-single emulation is added later.
- **`eigh` = symmetric/Hermitian** (values+vectors). Non-symmetric general `eig`
  is out of scope. **`svd`** is general rectangular with vectors.
- Jacobi is ~O(n³)·sweeps; ideal for moderate `n` and batched-small matrices.
  Very large `n` is future work (band-reduction route).
