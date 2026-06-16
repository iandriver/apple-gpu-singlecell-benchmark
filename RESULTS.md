# Results — Apple-GPU feasibility benchmark

Machine: **Apple M5 Pro, 20-core, 48 GB unified memory, Metal 4** · PyTorch 2.12 (MPS) · scanpy 1.12
Data: 50,000 cells × 20,000 genes sparse counts (67.6 M nonzeros, ~6.8% dense)

| Step | Type | CPU (scanpy/sklearn) | MPS compute-only | MPS incl. transfer | Verdict |
|---|---|---:|---:|---:|---|
| **normalize_total + log1p** | memory-bound (sparse) | 161 ms | 15 ms (**10.7×**) | 29 ms (**5.5×**) | GPU wins |
| **scale** (z-score + clip) | memory-bound (dense) | 157 ms | 28 ms (**5.7×**) | 34 ms (**4.6×**) | GPU wins |
| **PCA** (50 comps) | compute-bound | 321 ms | 230 ms (**1.4×**) | 240 ms (1.3×) | ~wash |
| **exact KNN** (k=15) | compute-bound | 581 ms | 6867 ms (**0.08×**) | 2277 ms (0.26×) | **GPU loses** |

Correctness checks passed (max |CPU − GPU| ≈ 1e-7 for normalize, 7e-5 for scale — both expected float32 rounding).

CPU Step-1 fairness: the 161 ms includes ~16 ms of AnnData construction + CSR copy. The pure
compute is ~145 ms (raw scipy `.data` path ~156 ms), so the ~10× compute win is genuine.

## The headline finding — it inverts the naive expectation

We expected: *memory-bound steps = no GPU win (shared unified bandwidth); compute-bound steps =
big GPU win.* The data says **almost the opposite**, and for an instructive reason:

1. **Memory-bound elementwise steps WON (5–10×)** — but largely because scanpy's CPU path is
   effectively **single-threaded** (`np.log1p`, sparse `.data` arithmetic don't use all 20 cores),
   while the GPU parallelizes massively over 67 M elements. So the win is real *today* but reflects
   an under-threaded CPU baseline, not a fundamental GPU advantage. A well-threaded CPU
   implementation would close much of this gap. **Absolute** savings are also small: ~150 ms → ~30 ms.

2. **The compute-bound steps — the ones we wanted the GPU for — are exactly where MPS fails:**
   - **PCA barely moved (1.4×)** because PyTorch-MPS has **no working GPU eigensolver/SVD/QR**:
     - `torch.pca_lowrank` / `torch.linalg.qr` (tall-skinny) → **hang indefinitely**
     - `torch.linalg.svd` → **not implemented on MPS → silent CPU fallback**
     - `torch.linalg.eigh` → **`NotImplementedError`**
     The only GPU-resident route is the hybrid here (Gram matmul on GPU + eig on **CPU**), and the
     CPU eig dominates, so the GPU contributes little.
   - **KNN LOST (4–12× slower)**: MPS `topk` over wide distance tiles is slow, and sklearn's CPU
     neighbors is well-optimized. (Production scanpy uses *approximate* neighbors via pynndescent,
     which is faster still — so exact-GPU loses even harder in practice.)

## What this means for the feasibility question

**The blocker is not unified-memory bandwidth, and not the choice of PyTorch vs CPU — it is the
missing GPU linear-algebra and graph ecosystem on Metal.**

- The steps that are *easy* to accelerate on MPS (elementwise normalize/scale) are the **least
  valuable** to port: they're already sub-200 ms and a threaded CPU build would rival the GPU.
- The steps that **dominate real single-cell runtime** — PCA, neighbor graph, UMAP, Leiden/Louvain
  clustering — are precisely the ones MPS **cannot do today** (no GPU SVD/eigh/QR; no graph library
  equivalent to cuGraph).

So a *useful* Apple-GPU rapids-singlecell built on **PyTorch-MPS is not feasible right now** for the
high-value operations. You could accelerate the cheap elementwise tier, but it wouldn't move the
needle on a real pipeline's wall-clock.

## The decisive question → tested, and it's settled: MLX doesn't help either

PyTorch-MPS lacks GPU eig/SVD. The hope was that **MLX** (Apple's own framework) — which *exposes*
`mlx.core.linalg.svd/qr/eigh` — would run them on the GPU. **Tested on MLX 0.31.2: it does not.**
All three raise:

```
ValueError: [linalg::svd]  This op is not yet supported on the GPU. Explicitly pass a CPU stream.
ValueError: [linalg::qr]   ... not yet supported on the GPU ...
ValueError: [linalg::eigh] ... not yet supported on the GPU ...
```

So **neither PyTorch-MPS nor MLX exposes an `eigh` / `svd` / `qr` on the Apple GPU today.** This is
not a framework-choice problem — it is a gap in the Apple-GPU numerical stack.

For a *general* GPU SVD/eig you would **hand-write Metal kernels** (e.g. a Jacobi eigensolver) or
build against Apple's lower-level Metal decomposition APIs — a bounded but specialized effort per
routine. Accelerate/LAPACK on Mac is CPU-only. **But for the low-rank PCA case the gap is routable
today without any kernel — see the Update.**

## Update — PCA *is* GPU-accelerable (correcting the bottom line)

The "PCA is a wash" result above used a *naive* algorithm (Gram matrix on GPU, eig on CPU), and the
CPU eig dominated. A better algorithm changes the verdict.

MPS lacks `eigh`/`svd`/`qr`, but it **does** ship `cholesky` (~5 ms, 2000×2000), `solve_triangular`,
and fast `matmul`. Those build a GPU QR via **Cholesky-QR** (`Q = Y · chol(YᵀY)⁻¹`), which builds a
**randomized SVD almost entirely on the GPU**; the only off-GPU step is a ~60×60 eig on the CPU
(microseconds). See [`pca_gpu_rsvd.py`](pca_gpu_rsvd.py).

| PCA route | CPU | Apple GPU | speedup |
|---|--:|--:|--:|
| naive (Gram + CPU eig) | 321 ms | 230 ms | 1.4× |
| **CholeskyQR randomized SVD** | 365 ms | **49 ms** | **7.5×** (6.6× incl. transfer) |

Leading singular values match sklearn to ~3e-4. Stability: Cholesky-QR squares the condition number,
so we run it twice (**CholeskyQR2**). Accuracy on the weakest of the 50 components is looser (~15%),
tightened with more oversampling / power iterations.

### Bottom line (revised)
- **PCA / truncated low-rank SVD: feasible on the Apple GPU today** (~1 day, no custom kernel, ~7.5×).
  The earlier "compute-bound is blocked" claim was too broad — algorithm choice, not the GPU, was
  the limiter.
- **Memory-bound elementwise steps** accelerate (5–10×) but are cheap in absolute terms.
- **Exact KNN, approximate neighbors, and graph clustering** (cuGraph-equivalents) remain genuinely
  missing on Metal; these are the real blockers to a full Apple-GPU rapids-singlecell.
- A **general drop-in GPU `svd`/`eigh`** still needs a Metal Jacobi kernel (bounded, multi-week),
  pluggable via `torch.mps.compile_shader()`. Revisit when the frameworks ship GPU linalg.
