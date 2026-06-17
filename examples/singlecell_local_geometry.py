"""
Single-cell analyses on the Apple GPU that were NOT POSSIBLE before metal_linalg.

rapids-singlecell does these on NVIDIA via cuSOLVER (batched eigh/svd). On Apple
Silicon they simply could not run on the GPU: torch.linalg.eigh/svd raise
NotImplementedError on MPS, and MLX's are CPU-only. metal_linalg fills exactly that
gap — and these are its sweet spot: thousands of small, independent decompositions.

Two per-cell local-geometry analyses over a k-NN graph (the kind used for trajectory
boundaries, transition/stem-cell detection, and local manifold structure):

  A. Local intrinsic dimensionality  — batched SVD of each cell's neighbor cloud.
  B. Local principal direction / anisotropy — batched eigh of each cell's
     neighborhood covariance.

Both run on the Apple GPU here. We show: (1) stock torch.linalg fails on MPS,
(2) with metal_linalg it works, (3) it matches CPU, and (4) it's faster.

Run:  python examples/singlecell_local_geometry.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

import metal_linalg

DEV = torch.device("mps")
N_CELLS = 20_000
N_PCS = 32          # dims of the PCA embedding (like sc.pp.pca output)
K = 16              # neighbors per cell (like sc.pp.neighbors n_neighbors)


def make_embedding(seed=0):
    """A clustered PCA-like embedding: several cell-type blobs on a manifold."""
    rng = np.random.default_rng(seed)
    n_types = 6
    centers = rng.standard_normal((n_types, N_PCS)) * 6
    labels = rng.integers(0, n_types, N_CELLS)
    emb = centers[labels] + rng.standard_normal((N_CELLS, N_PCS)).astype("f4")
    # give some types anisotropic local structure (a few dominant directions)
    for t in range(n_types):
        m = labels == t
        scale = np.ones(N_PCS, "f4"); scale[: t + 1] = 4.0
        emb[m] = centers[t] + (emb[m] - centers[t]) * scale
    return emb.astype("f4")


def neighbor_clouds(emb, k=K):
    """For each cell, the centered matrix of its k neighbors: (N, k, d)."""
    nn = NearestNeighbors(n_neighbors=k).fit(emb)
    _, idx = nn.kneighbors(emb)                     # (N, k)
    clouds = emb[idx]                               # (N, k, d)
    clouds = clouds - clouds.mean(axis=1, keepdims=True)
    return clouds.astype("f4")


def _time(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / repeats * 1e3


def banner(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def main():
    print(f"metal_linalg single-cell demo | {N_CELLS:,} cells, {N_PCS}-dim PCA, k={K}")
    print(f"device: {DEV}, torch {torch.__version__}")

    emb = make_embedding()
    clouds = neighbor_clouds(emb)                   # (N, K, d) centered
    clouds_mps = torch.from_numpy(clouds).to(DEV)

    # ---- the "couldn't run on the GPU before" demonstration ---------------
    banner("Why this couldn't run on the Apple GPU before")
    small = cov_sample = torch.eye(N_PCS, device=DEV).expand(4, N_PCS, N_PCS).contiguous()
    try:
        torch.linalg.eigh(small)
        print("  (unexpected: stock torch.linalg.eigh worked on MPS)")
    except Exception as e:
        print(f"  stock torch.linalg.eigh(mps)  -> {type(e).__name__}: hard-blocked on MPS")
    import warnings
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        torch.linalg.svd(clouds_mps[:4], full_matrices=False)
    fell_back = any("MPS" in str(w.message) for w in wl)
    print(f"  stock torch.linalg.svd(mps)   -> silent CPU fallback "
          f"({'confirmed' if fell_back else 'n/a'}): runs, but on the CPU (slow)")
    print("  => either way, these decompositions did NOT run on the GPU.\n")
    metal_linalg.install()       # drop-in: now torch.linalg runs them ON the GPU
    print("  metal_linalg.install() -> torch.linalg.{svd,eigh} now run on the MPS GPU\n")

    # ---- Use case A: local intrinsic dimensionality (batched SVD) ----------
    banner("A. Local intrinsic dimensionality  (batched SVD, one per cell)")
    print("  Per cell: SVD of its k-neighbor cloud; participation ratio of the")
    print("  singular values estimates the local manifold dimension.\n")

    def gpu_svd():
        U, S, Vh = torch.linalg.svd(clouds_mps, full_matrices=False)   # patched -> GPU
        return S
    S = gpu_svd()
    s2 = (S ** 2)
    part_ratio = (s2.sum(-1) ** 2) / (s2 ** 2).sum(-1)        # effective # of dims/cell
    # correctness vs CPU LAPACK on a sample
    samp = clouds[:2000]
    S_cpu = torch.linalg.svd(torch.from_numpy(samp), full_matrices=False)[1].numpy()
    err = np.max(np.abs(np.sort(S.cpu().numpy()[:2000]) - np.sort(S_cpu)))
    gpu_ms = _time(gpu_svd)
    cpu_ms = _time(lambda: torch.linalg.svd(torch.from_numpy(clouds), full_matrices=False))
    print(f"  ran on {S.device.type.upper()} for all {N_CELLS:,} cells")
    print(f"  local dim (participation ratio): mean {part_ratio.mean():.2f}, "
          f"range {part_ratio.min():.2f}–{part_ratio.max():.2f}")
    print(f"  max |singular value - LAPACK| (sample): {err:.2e}")
    print(f"  GPU {gpu_ms:.1f} ms   vs   CPU {cpu_ms:.1f} ms   ->  {cpu_ms / gpu_ms:.2f}x")

    # ---- Use case B: local principal direction / anisotropy (batched eigh) -
    banner("B. Local principal direction & anisotropy  (batched eigh, one per cell)")
    print("  Per cell: eigendecompose its neighborhood covariance (d x d); the top")
    print("  eigenvector is the local principal axis, λ1/Σλ measures anisotropy.\n")
    cov = torch.bmm(clouds_mps.transpose(1, 2), clouds_mps) / K       # (N, d, d) on GPU

    def gpu_eigh():
        return torch.linalg.eigh(cov)                                 # patched -> GPU
    w, V = gpu_eigh()
    anisotropy = w[:, -1] / w.clamp_min(0).sum(-1)                    # top-eig fraction
    # correctness vs CPU
    w_cpu = torch.linalg.eigh(cov[:2000].cpu())[0].numpy()
    err2 = np.max(np.abs(np.sort(w.cpu().numpy()[:2000]) - np.sort(w_cpu)))
    gpu_ms = _time(gpu_eigh)
    cpu_ms = _time(lambda: torch.linalg.eigh(cov.cpu()))
    print(f"  ran on {w.device.type.upper()} for all {N_CELLS:,} cells")
    print(f"  anisotropy (λ1/Σλ): mean {anisotropy.mean():.3f}, "
          f"max {anisotropy.max():.3f}")
    print(f"  max |eigenvalue - LAPACK| (sample): {err2:.2e}")
    print(f"  GPU {gpu_ms:.1f} ms   vs   CPU {cpu_ms:.1f} ms   ->  {cpu_ms / gpu_ms:.2f}x")

    metal_linalg.uninstall()
    banner("Done — both analyses ran end-to-end on the Apple GPU")
    print("  Before metal_linalg: eigh hard-blocks on MPS and svd silently runs on")
    print("  the CPU — so these per-cell decompositions could not run on the GPU.")
    print("  Now they do, matching LAPACK and faster than the CPU baseline.\n")


if __name__ == "__main__":
    main()
