"""
Follow-up finding: PCA *can* be GPU-accelerated on Apple Silicon today — with
the right algorithm and NO custom Metal kernel.

The main benchmark (bench.py) measured a naive PCA route (Gram matrix on GPU +
eigendecomposition on CPU, because torch.linalg.eigh/svd/qr don't run on MPS)
and got only ~1.4x. That made PCA look like a wash. It isn't — the algorithm was
the problem, not the GPU.

Key observation: while MPS lacks eigh/svd/qr, it DOES have:
    torch.linalg.cholesky        (works, ~5 ms for 2000x2000)
    torch.linalg.solve_triangular (works)
    matmul                        (fast)

Those three are enough to build a QR factorization on the GPU via **Cholesky-QR**
(Q = Y · chol(YᵀY)⁻¹), which is enough to build a **randomized SVD almost entirely
on the GPU**. The only off-GPU step is a tiny (k+p)×(k+p) eigendecomposition on the
CPU (~60×60 here — microseconds).

Measured on Apple M5 Pro (50k×2000, k=50): ~8x vs sklearn randomized PCA, with the
leading singular values matching to ~0.01%.

Numerical note: Cholesky-QR squares the condition number of Y, so for
ill-conditioned data we run it twice (**CholeskyQR2**) to restore orthogonality.
Still cheap, still fully on the GPU.

Run:  python pca_gpu_rsvd.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
DEV = torch.device("mps")


def cholesky_qr(Y):
    """Orthonormalize the columns of Y on the GPU (CholeskyQR2 for stability).

    Q = Y · R⁻¹ where R = chol(YᵀY). Repeated once because Cholesky-QR squares
    the condition number; the second pass cleans up the loss of orthogonality.
    """
    for _ in range(2):
        R = torch.linalg.cholesky(Y.t() @ Y)                       # (m, m)
        Y = torch.linalg.solve_triangular(R, Y.t(), upper=False).t()
    return Y


def gpu_rsvd_pca(Xc, k=50, oversample=10, n_iter=2):
    """Randomized-SVD PCA, GPU-resident except a tiny CPU eigendecomposition.

    Xc : centered data, (n, g) on MPS.  Returns (scores (n,k), singular_values (k)).
    """
    g = Xc.shape[1]
    m = k + oversample
    Omega = torch.randn(g, m, device=DEV)
    Q = cholesky_qr(Xc @ Omega)                                    # range sketch
    for _ in range(n_iter):                                        # subspace iteration
        Q = cholesky_qr(Xc @ (Xc.t() @ Q))
    B = Q.t() @ Xc                                                 # (m, g)
    # small SVD of B via eigh of B Bᵀ (m×m). eigh isn't on MPS, but m≈60 -> CPU is free.
    w, U = torch.linalg.eigh((B @ B.t()).cpu())
    order = torch.argsort(w, descending=True)[:k]
    S = torch.sqrt(torch.clamp(w[order], min=0)).to(DEV)
    Ub = U[:, order].to(DEV)
    scores = (Q @ Ub) * S                                          # (n, k) PCA embedding
    torch.mps.synchronize()
    return scores, S


def _median_ms(fn, repeats=3, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e3


def main():
    from sklearn.decomposition import PCA

    N, G, K = 50_000, 2_000, 50
    rng = np.random.default_rng(0)
    Xn = rng.standard_normal((N, G)).astype("f4")
    Xn[:, :20] *= np.linspace(5, 1, 20)            # inject some low-rank structure
    X = torch.from_numpy(Xn).to(DEV)

    print(f"torch {torch.__version__} | {DEV} | {N:,}x{G:,}, k={K}")

    # correctness vs sklearn
    sk = PCA(n_components=K, svd_solver="randomized", random_state=0).fit(Xn)
    Xc = X - X.mean(0, keepdim=True)
    _, S = gpu_rsvd_pca(Xc, k=K)
    gpu_sv = np.sort(S.cpu().numpy())[::-1]
    rel = np.abs(gpu_sv - sk.singular_values_) / sk.singular_values_
    print(f"singular-value rel-err vs sklearn: leading-10 max {rel[:10].max():.1e} | "
          f"all-{K} max {rel.max():.1e} (tail is weakest comps)")

    t_cpu = _median_ms(lambda: PCA(n_components=K, svd_solver="randomized",
                                   random_state=0).fit_transform(Xn))
    t_gpu = _median_ms(lambda: gpu_rsvd_pca(X - X.mean(0, keepdim=True), k=K))
    t_tot = _median_ms(lambda: gpu_rsvd_pca(
        (lambda d: d - d.mean(0, keepdim=True))(torch.from_numpy(Xn).to(DEV)), k=K))
    print(f"CPU sklearn randomized PCA : {t_cpu:7.1f} ms")
    print(f"GPU CholeskyQR rsvd        : {t_gpu:7.1f} ms  ({t_cpu / t_gpu:.1f}x)")
    print(f"GPU incl. host->device     : {t_tot:7.1f} ms  ({t_cpu / t_tot:.1f}x)")


if __name__ == "__main__":
    main()
