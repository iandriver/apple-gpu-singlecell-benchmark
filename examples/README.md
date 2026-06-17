# Examples — single-cell analyses unlocked on the Apple GPU

These run single-cell operations **on the Apple Silicon GPU that previously could
not run there**, using [`metal_linalg`](../metal_linalg). rapids-singlecell performs
the equivalent steps on NVIDIA via cuSOLVER's batched eigh/svd; on Apple GPUs those
were unavailable — `torch.linalg.eigh` hard-blocks on MPS (`NotImplementedError`)
and `torch.linalg.svd` silently falls back to the CPU. `metal_linalg`'s batched
Jacobi kernels fill exactly that gap, and per-cell decompositions are its sweet spot
(thousands of small, independent problems).

## [`singlecell_local_geometry.py`](singlecell_local_geometry.py)

Per-cell **local manifold geometry** over a k-NN graph — the kind of computation used
for trajectory boundaries, transition/stem-cell detection, and local structure. On
20,000 cells (32-dim PCA embedding, k=16), measured on an Apple M5 Pro:

| Analysis | kernel | runs on | vs CPU |
|---|---|---|---|
| **A. Local intrinsic dimensionality** — SVD of each cell's neighbor cloud; participation ratio of the singular values estimates the local manifold dimension | batched SVD | **Apple GPU** | **5.9×** |
| **B. Local principal direction / anisotropy** — eigendecompose each cell's neighborhood covariance; top eigenvector = local principal axis, λ1/Σλ = anisotropy | batched eigh | **Apple GPU** | **3.0×** |

Both match LAPACK to ~1.5e-5. The script first demonstrates that stock
`torch.linalg.eigh`/`svd` can't do this on the GPU, then runs it on the GPU via the
`metal_linalg.install()` drop-in patch.

```bash
pip install -e .            # from the repo root
python examples/singlecell_local_geometry.py
```

### Where this fits in a pipeline
These are the GPU-friendly *batched-small* members of the single-cell linear-algebra
family. The heavy single-matrix step — **PCA of the full cell×gene matrix** — is
covered separately by the CholeskyQR randomized SVD in
[`../pca_gpu_rsvd.py`](../pca_gpu_rsvd.py) (~7.5× on the Apple GPU, no custom kernel).
Together they cover the decomposition-bound steps of a scanpy/rapids-singlecell-style
workflow on Apple Silicon.
