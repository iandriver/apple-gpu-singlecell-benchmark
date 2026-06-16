"""
Apple-GPU (Metal/MPS) feasibility benchmark for single-cell preprocessing.

Goal
----
Measure, on *this* Mac, whether moving the core scanpy preprocessing steps onto
the Apple GPU (via PyTorch's MPS backend) is actually faster than well-threaded
CPU code -- and *which* steps are worth porting.

The central physics fact this benchmark exists to expose:

    Apple Silicon has UNIFIED memory. The CPU and GPU share one pool of RAM and
    one memory-bandwidth budget. On a discrete NVIDIA card the GPU has its own
    fast VRAM, so memory-bound work wins big by moving to the GPU. On a Mac it
    does not automatically win, because both processors draw from the same pipe.

So we expect:
  * MEMORY-BOUND steps (normalize, log1p, scale): GPU ~ CPU, modest or no win.
  * COMPUTE-BOUND steps (PCA via SVD, exact KNN distances): GPU should win.

Each operation is implemented twice and timed with proper warm-up and GPU
synchronization. We also report the cost of the host->device transfer
separately, because in a CPU-fallback design (Option B) you pay that on every
hand-off to the GPU.

This file is standalone: it does NOT import rapids_singlecell and does not touch
the package. Run:  ./.venv/bin/python bench.py
"""

from __future__ import annotations

import signal
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config -- a realistic small/medium single-cell workflow.
# ---------------------------------------------------------------------------
SEED = 0
N_CELLS = 50_000          # observations (rows)
N_GENES = 20_000          # features (cols) -- the full matrix is sparse
DENSITY = 0.07            # ~7% nonzero, typical for scRNA-seq counts
N_HVG = 2_000             # genes kept after highly-variable selection
N_PCS = 50                # PCA components
K_NEIGHBORS = 15          # KNN graph degree
TARGET_SUM = 1e4          # normalize_total target (counts-per-10k)
N_REPEATS = 5             # timed repeats; we report the median

DEVICE = torch.device("mps")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _sync():
    """Block until all queued MPS work is finished.

    MPS dispatches kernels asynchronously: the Python call returns before the
    GPU is done. Without this barrier we would time the *enqueue*, not the
    *compute*, and report fantasy speedups.
    """
    torch.mps.synchronize()


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def timeit(fn, repeats=N_REPEATS, warmup=2, watchdog_s=90):
    """Run fn() warmup times (untimed), then `repeats` times; return median seconds.

    Warm-up matters on MPS: the first launch of a kernel pays a one-time
    shader-compilation / graph-capture cost that we do not want in the steady
    state number.

    A SIGALRM watchdog guards every call: some MPS linalg kernels (QR, SVD)
    hang indefinitely, and we never want one to lock up the whole benchmark.
    A hung op returns NaN seconds instead of stalling forever.
    """
    signal.signal(signal.SIGALRM, _alarm)
    try:
        for _ in range(warmup):
            signal.alarm(watchdog_s)
            fn()
        times = []
        for _ in range(repeats):
            signal.alarm(watchdog_s)
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.median(times))
    except _Timeout:
        print(f"    !! call exceeded {watchdog_s}s watchdog -- treating as hang")
        return float("nan")
    finally:
        signal.alarm(0)


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def report(name, cpu_s, gpu_compute_s, gpu_total_s, err=None):
    """Pretty-print one comparison row.

    gpu_compute_s = kernel time only (data already on GPU)
    gpu_total_s   = kernel time + host->device transfer (the real cost on a
                    per-call hand-off from a CPU pipeline)
    """
    sp_compute = cpu_s / gpu_compute_s if gpu_compute_s else float("nan")
    sp_total = cpu_s / gpu_total_s if gpu_total_s else float("nan")
    print(f"\n  {name}")
    print(f"    CPU (scanpy/sklearn)      : {cpu_s * 1e3:9.1f} ms")
    print(f"    MPS compute-only          : {gpu_compute_s * 1e3:9.1f} ms   ({sp_compute:5.2f}x vs CPU)")
    print(f"    MPS incl. host->device    : {gpu_total_s * 1e3:9.1f} ms   ({sp_total:5.2f}x vs CPU)")
    if err is not None:
        print(f"    max|CPU - GPU| (correctness): {err:.3e}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_counts():
    """A reproducible sparse count matrix that looks like scRNA-seq.

    Nonzero entries are Poisson-distributed small integers; the layout is CSR
    (cells x genes), exactly what AnnData hands you.
    """
    rng = np.random.default_rng(SEED)
    nnz = int(N_CELLS * N_GENES * DENSITY)
    rows = rng.integers(0, N_CELLS, size=nnz)
    cols = rng.integers(0, N_GENES, size=nnz)
    vals = rng.poisson(2.0, size=nnz).astype(np.float32) + 1.0
    X = sp.csr_matrix((vals, (rows, cols)), shape=(N_CELLS, N_GENES))
    X.sum_duplicates()
    return X


# ===========================================================================
# STEP 1: normalize_total + log1p   --- MEMORY-BOUND, operates on nonzeros
# ===========================================================================
# Both normalize_total and log1p only ever touch the *nonzero* values of the
# matrix (zeros stay zero through both). The honest GPU implementation therefore
# moves only the CSR `.data` array (the nonzeros) to the GPU -- NOT a dense
# expansion. This is the fair sparse-vs-sparse comparison.
#
# normalize_total divides each nonzero by its cell's total count, times a target.
# We compute per-cell totals on the GPU with a segmented sum (scatter_add).
def cpu_normalize_log1p(X):
    import scanpy as sc
    from anndata import AnnData

    adata = AnnData(X.copy())
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)
    return adata.X


def gpu_prep_normalize(X):
    """Move the sparse structure to the GPU once. Returns tensors + a builder."""
    data = torch.from_numpy(X.data.astype(np.float32))
    indptr = torch.from_numpy(X.indptr.astype(np.int64))
    row_lengths = (indptr[1:] - indptr[:-1])
    # row id for every nonzero, e.g. [0,0,0,1,1,2,...]
    row_ids = torch.repeat_interleave(torch.arange(N_CELLS), row_lengths)
    return data, row_ids


def gpu_normalize_log1p(data_dev, row_ids_dev):
    # per-cell total = segmented sum of nonzeros by row
    totals = torch.zeros(N_CELLS, device=DEVICE, dtype=torch.float32)
    totals.scatter_add_(0, row_ids_dev, data_dev)
    factor = TARGET_SUM / totals
    out = data_dev * factor[row_ids_dev]
    out = torch.log1p(out)
    _sync()
    return out


# ===========================================================================
# STEP 2: scale (z-score per gene + clip)  --- MEMORY-BOUND, dense
# ===========================================================================
# Scaling centers each gene to mean 0: subtracting the mean turns every zero
# into a nonzero, so the matrix becomes dense. In a real pipeline this runs on
# the HVG-subset (2000 genes), so we operate on a dense 50k x 2000 block.
def cpu_scale(dense):
    import scanpy as sc

    arr = dense.copy()
    sc.pp.scale(arr, max_value=10.0)   # scanpy works on the ndarray in place-ish
    return arr


def gpu_scale(X_dev):
    mean = X_dev.mean(dim=0, keepdim=True)
    std = X_dev.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    out = (X_dev - mean) / std
    out = torch.clamp(out, max=10.0)
    _sync()
    return out


# ===========================================================================
# STEP 3: PCA (truncated SVD)  --- COMPUTE-BOUND
# ===========================================================================
# This is where the GPU should earn its keep: SVD is O(n * g * k) flops, not
# just a memory sweep.
def cpu_pca(dense):
    from sklearn.decomposition import PCA

    return PCA(n_components=N_PCS, svd_solver="randomized", random_state=SEED).fit_transform(dense)


def gpu_pca(X_dev):
    # IMPORTANT FINDING: PyTorch-MPS cannot do PCA the normal way today.
    #   torch.pca_lowrank / torch.linalg.qr  -> HANG on MPS (tall-skinny QR)
    #   torch.linalg.svd                     -> not implemented, silent CPU fallback
    #   torch.linalg.eigh                    -> NotImplementedError on MPS
    # The only way to keep the *heavy* work on the GPU is the Gram-matrix route:
    # the big O(n * g^2) matmul runs on the GPU, and the small (g x g)
    # eigendecomposition is done on the CPU (g=2000 -> cheap). This hybrid is
    # the realistic best-case for PCA on Apple GPU with today's PyTorch.
    Xc = X_dev - X_dev.mean(dim=0, keepdim=True)
    C = Xc.t() @ Xc                       # (g, g) Gram matrix -- heavy matmul on GPU
    _sync()
    w, V = torch.linalg.eigh(C.cpu())     # eig on CPU (no MPS kernel exists)
    V = V[:, -N_PCS:].to(DEVICE)          # top-N_PCS eigenvectors
    emb = Xc @ V                          # project back on GPU
    _sync()
    return emb


# ===========================================================================
# STEP 4: exact KNN distances  --- COMPUTE-BOUND
# ===========================================================================
# Production scanpy uses APPROXIMATE neighbors (pynndescent); to compare the
# raw GPU vs CPU distance kernel fairly we run EXACT brute-force on both sides
# and tile the GPU side so the 50k x 50k distance block never fully materializes.
def cpu_knn(emb):
    from sklearn.neighbors import NearestNeighbors

    # algorithm="auto" lets sklearn pick a tree/chunked path instead of
    # materializing the full 50k x 50k distance matrix (which can OOM).
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="auto", metric="euclidean")
    nn.fit(emb)
    dist, idx = nn.kneighbors(emb)
    return idx


def gpu_knn(emb_dev, tile=4096):
    n = emb_dev.shape[0]
    out_idx = torch.empty((n, K_NEIGHBORS), dtype=torch.long, device=DEVICE)
    for start in range(0, n, tile):
        block = emb_dev[start:start + tile]               # (t, d)
        d = torch.cdist(block, emb_dev)                   # (t, n) distances
        _, idx = torch.topk(d, K_NEIGHBORS, dim=1, largest=False)
        out_idx[start:start + idx.shape[0]] = idx
    _sync()
    return out_idx


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE} | torch {torch.__version__}")
    print(f"Matrix: {N_CELLS:,} cells x {N_GENES:,} genes @ {DENSITY:.0%} density")

    banner("Building data")
    t = time.perf_counter()
    X = make_counts()
    print(f"  sparse counts: nnz={X.nnz:,}  ({X.nnz / (N_CELLS * N_GENES):.1%} dense)  "
          f"built in {time.perf_counter() - t:.1f}s")

    # ---- STEP 1: normalize + log1p (sparse, memory-bound) -----------------
    banner("STEP 1  normalize_total + log1p   [memory-bound, sparse .data only]")
    cpu_s = timeit(lambda: cpu_normalize_log1p(X))
    data_dev = torch.from_numpy(X.data.astype(np.float32)).to(DEVICE)
    data_h, row_ids_h = gpu_prep_normalize(X)

    def gpu_call():
        gpu_normalize_log1p(data_dev, row_ids_dev)
    row_ids_dev = row_ids_h.to(DEVICE)
    gpu_compute_s = timeit(gpu_call)

    def gpu_total_call():
        d = data_h.to(DEVICE)
        r = row_ids_h.to(DEVICE)
        gpu_normalize_log1p(d, r)
    gpu_total_s = timeit(gpu_total_call)

    cpu_res = cpu_normalize_log1p(X)
    gpu_res = gpu_normalize_log1p(data_dev, row_ids_dev).cpu().numpy()
    err = float(np.max(np.abs(np.sort(cpu_res.data) - np.sort(gpu_res))))
    report("normalize_total + log1p", cpu_s, gpu_compute_s, gpu_total_s, err)

    # Build the dense HVG block used by the next 3 steps. We pick the 2000
    # highest-variance genes from the normalized matrix (a stand-in for HVG).
    banner("Preparing HVG-subset dense block for steps 2-4")
    Xln = cpu_res
    gene_var = np.asarray(Xln.power(2).mean(axis=0)).ravel() - np.asarray(Xln.mean(axis=0)).ravel() ** 2
    hvg = np.argsort(gene_var)[-N_HVG:]
    dense = np.asarray(Xln[:, hvg].todense(), dtype=np.float32)
    print(f"  dense block: {dense.shape[0]:,} x {dense.shape[1]:,}  "
          f"({dense.nbytes / 1e6:.0f} MB float32)")

    # ---- STEP 2: scale (dense, memory-bound) ------------------------------
    banner("STEP 2  scale (z-score per gene + clip)   [memory-bound, dense]")
    cpu_s = timeit(lambda: cpu_scale(dense))
    X_dev = torch.from_numpy(dense).to(DEVICE)
    gpu_compute_s = timeit(lambda: gpu_scale(X_dev))
    gpu_total_s = timeit(lambda: gpu_scale(torch.from_numpy(dense).to(DEVICE)))
    cpu_res2 = cpu_scale(dense)
    gpu_res2 = gpu_scale(X_dev).cpu().numpy()
    err = float(np.max(np.abs(cpu_res2 - gpu_res2)))
    report("scale", cpu_s, gpu_compute_s, gpu_total_s, err)

    scaled = gpu_res2  # use the scaled block downstream

    # ---- STEP 3: PCA (compute-bound) --------------------------------------
    banner("STEP 3  PCA   [COMPUTE-bound; GPU path = Gram matmul on GPU + eig on CPU]")
    cpu_s = timeit(lambda: cpu_pca(scaled), repeats=3)
    Xs_dev = torch.from_numpy(scaled).to(DEVICE)
    gpu_compute_s = timeit(lambda: gpu_pca(Xs_dev), repeats=3)
    gpu_total_s = timeit(lambda: gpu_pca(torch.from_numpy(scaled).to(DEVICE)), repeats=3)
    report("PCA (n_comps=%d, hybrid)" % N_PCS, cpu_s, gpu_compute_s, gpu_total_s)
    emb = gpu_pca(Xs_dev).cpu().numpy()

    # ---- STEP 4: exact KNN (compute-bound) --------------------------------
    banner("STEP 4  exact brute-force KNN   [COMPUTE-bound]")
    cpu_s = timeit(lambda: cpu_knn(emb), repeats=3)
    emb_dev = torch.from_numpy(emb).to(DEVICE)
    gpu_compute_s = timeit(lambda: gpu_knn(emb_dev), repeats=3)
    gpu_total_s = timeit(lambda: gpu_knn(torch.from_numpy(emb).to(DEVICE)), repeats=3)
    report("KNN (k=%d, exact)" % K_NEIGHBORS, cpu_s, gpu_compute_s, gpu_total_s)

    banner("Done")
    print("  Read the per-step speedups against the memory-bound vs compute-bound")
    print("  labels. That contrast is the whole feasibility story.\n")


if __name__ == "__main__":
    main()
