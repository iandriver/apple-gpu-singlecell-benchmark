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
| **2** | **batched eigh — one threadgroup/matrix, threadgroup-resident: actual GPU win** | ✅ **done — up to 7.5×** |
| **2b** | **batched eigh for larger n (32<n≤64), V in device memory** | ✅ **done — 3.1× @48, 1.3× @64** |
| **3** | **batched svd — one-sided Jacobi (tall/wide), with vectors** | ✅ **done — up to 6.4×** |
| **4** | **robustness (auto-dispatch + CPU fallback) & precision (fp16 tested)** | ✅ **done** |
| 5 | integration: patch `torch.linalg.{svd,eigh}` on MPS, packaging | planned |

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
comes from batching (Phase 2). The go/no-go was **correctness — passed**.

## What Phase 2 delivered — the actual GPU speedup

`batched_jacobi_eigh`: **one threadgroup per matrix**, the whole grid running B
matrices at once, each kept entirely in **threadgroup memory** (one global load in,
one store out — all Jacobi sweeps run on-chip). This is the throughput regime the
GPU owns: many small independent eigendecompositions.

Measured ([`test_phase2.py`](test_phase2.py)), GPU `batched_eigh` vs CPU/Accelerate
batched `torch.linalg.eigh`, correctness ~1e-6 per matrix (with the occupancy
tuning below):

| n | batch | GPU ms | CPU ms | speedup |
|--:|--:|--:|--:|--:|
| 16 | 4,096 | 6.8 | 42.2 | **6.2×** |
| 16 | 16,384 | 21.4 | 159.9 | **7.5×** |
| 32 | 1,024 | 8.1 | 37.6 | **4.7×** |
| 32 | 16,384 | 113.0 | 551.1 | **4.9×** |

GPU wins at every tested point (down to batch=256).

**Occupancy tuning** ([`tune_batched.py`](tune_batched.py)): the kernel is
parameterized in threadgroup-memory footprint (`BATCH_MAX_BN`, sized to n) and
threads-per-matrix (`BATCH_BTG`); compiling a specialization per size lets more
matrices stay resident per core. Tuning lifted n=16 from 6.2× to **7.5×** and
n=32 from ~3.9× to ~4.9×. The measured optima (e.g. fewer threads/matrix:
btg=32) are the baked-in defaults.

Caveats (honest): (1) the speedup is the *steady-state throughput ratio* — both
sides scale linearly with batch once saturated, so it's not unbounded; (2) timings
are **data-resident-on-GPU** (the realistic case when eigh is one step in a GPU
pipeline), matching the "compute-only" column in the parent benchmark.

### eigh speedup vs n (batch ≥ 4096)

| n | 8 | 16 | 32 | 48 | 64 |
|---|--:|--:|--:|--:|--:|
| GPU speedup vs CPU | 3.7× | **7.5×** | 4.9× | 3.1× | 1.3× |

n ≤ 32 is fully threadgroup-resident; 32 < n ≤ 64 keeps A on-chip and V in device
memory (`batched_jacobi_eigh_vg`). The win **peaks around n=16** and narrows past
n=32 — by n=64 it's marginal (V-in-global traffic, lower occupancy, and Jacobi's
growing flop disadvantage). The batched sweet spot is small matrices, which is
exactly where batching is the right tool.

## What Phase 3 delivered — batched SVD

`batched_jacobi_svd`: one-sided (Hestenes) Jacobi, one threadgroup per matrix,
columns orthogonalized in threadgroup memory. Gives U, S, **and V** with good
small-singular-value accuracy (unlike a Gram/AᵀA shortcut). Tall handled directly;
wide via transpose. Correctness vs LAPACK ~1e-6 (recon, U/V orthonormality, values).

Speed ([`test_phase3.py`](test_phase3.py)) vs CPU/Accelerate batched `torch.linalg.svd`:

| shape | batch | GPU ms | CPU ms | speedup |
|---|--:|--:|--:|--:|
| 48×16 | 4,096 | 15.6 | 96.0 | **6.1×** |
| 48×16 | 16,384 | 60.3 | 382.7 | **6.4×** |
| 64×32 | 4,096 | 69.5 | 263.8 | **3.8×** |
| 64×32 | 16,384 | 275.6 | 1055.5 | **3.8×** |

## What Phase 4 delivered — robustness, and an fp16 dead end

**Auto-dispatch** ([`dispatch.py`](dispatch.py)): `eigh(A)` / `svd(A)` route batched
small matrices to the fast GPU kernels and everything else (single matrices, n>64)
to CPU LAPACK — so callers get the speedup automatically and a correct answer
always. A `verify=True` guard recomputes the residual and falls back to CPU for any
matrix that didn't converge (demonstrated: an under-converged batch at residual
0.55 is caught and corrected to 1.6e-6).

**fp16 — measured, rejected.** A half-storage variant (`store="fp16"`, compute
still fp32) was built and benchmarked. It is **not** a speedup on this GPU:

| n | fp32 ms | fp16 ms | fp16/fp32 | fp16 recon |
|--:|--:|--:|--:|--:|
| 16 | 22.1 | 31.2 | **0.71×** (slower) | 5e-3 |
| 32 | 113.1 | 102.4 | 1.10× | 7e-3 |

On M-series the fp16 ALU runs at ~fp32 rate, and the halved threadgroup storage
didn't buy enough occupancy to win — while accuracy drops to ~fp16. Conclusion:
**keep fp32.** (The capability stays in the kernel, off by default, as documented
evidence.)

## Files

| File | What |
|---|---|
| [`kernels.metal`](kernels.metal) | Metal source. P0: `saxpy`, `apply_col_rotation`. P1: `jacobi_eigh`. P2: `batched_jacobi_eigh`(+`_vg` for large n). P3: `batched_jacobi_svd`. |
| [`_dispatch.py`](_dispatch.py) | `compile_shader` singleton + MPS-tensor dispatch helpers. |
| [`kernels.py`](kernels.py) | Python wrappers + `metal_eigh`/`metal_svd` entry points. |
| [`reference.py`](reference.py) | Accuracy metrics + pathological test matrices. |
| [`bench.py`](bench.py) | CPU-baseline benchmark harness. |
| [`test_phase0.py`](test_phase0.py) | Phase 0 acceptance test (13 checks). |
| [`test_phase1.py`](test_phase1.py) | Phase 1 acceptance test: GPU Jacobi eigh vs LAPACK (14 checks). |
| [`test_phase2.py`](test_phase2.py) | Phase 2: batched eigh correctness + GPU-vs-CPU speed. |
| [`test_phase2b.py`](test_phase2b.py) | Phase 2b: batched eigh at n=48/64 (V-in-global). |
| [`test_phase3.py`](test_phase3.py) | Phase 3: batched SVD correctness (tall/wide) + speed. |
| [`tune_batched.py`](tune_batched.py) | Occupancy autotune sweep for batched eigh. |
| [`dispatch.py`](dispatch.py) | Phase 4: `eigh`/`svd` auto-dispatch (GPU batched ↔ CPU) + verify-fallback. |
| [`test_phase4.py`](test_phase4.py) | Phase 4: fp16 measurement, dispatch routing, fallback guard. |

## Run

```bash
# from the repo root, using the project venv
python -m metal_linalg.test_phase0    # integration scaffold  -> 13 passed
python -m metal_linalg.test_phase1    # GPU Jacobi eigh vs LAPACK -> 14 passed
python -m metal_linalg.test_phase2    # batched eigh: correctness + speed (up to 7.5x)
python -m metal_linalg.test_phase2b   # batched eigh at n=48/64
python -m metal_linalg.test_phase3    # batched SVD: correctness + speed (up to 6.4x)
python -m metal_linalg.test_phase4    # fp16 (rejected) + auto-dispatch + fallback guard
python -m metal_linalg.tune_batched   # occupancy autotune
```

Usable API (auto-routes to the fast GPU path, CPU fallback otherwise):

```python
from metal_linalg import eigh, svd
w, V   = eigh(A)            # A: (B,n,n), n<=64 -> GPU; else CPU
U, S, Vh = svd(A)           # A: (B,m,n), small -> GPU; else CPU
w, V   = eigh(A, verify=True)   # recompute-on-CPU guard for any non-converged matrix
```

## Scope notes

- **fp32 primary** (fp16 opportunistic). Apple GPUs have no native fp64 — that's
  the precision ceiling unless double-single emulation is added later.
- **`eigh` = symmetric/Hermitian** (values+vectors). Non-symmetric general `eig`
  is out of scope. **`svd`** is general rectangular with vectors.
- Jacobi is ~O(n³)·sweeps; ideal for moderate `n` and batched-small matrices.
  Very large `n` is future work (band-reduction route).
