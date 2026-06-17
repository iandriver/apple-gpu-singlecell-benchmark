"""
Accuracy harness: metrics and pathological test matrices used to validate any
eigh/svd implementation against LAPACK. Phase 1 kernels are graded with exactly
these functions, so the bar is defined now.
"""

from __future__ import annotations

import numpy as np

# fp32 machine epsilon ~1.2e-7. A correct fp32 solver should land within a small
# multiple of eps * n for well-conditioned inputs; we use generous-but-meaningful
# tolerances and report the raw numbers so regressions are visible.
FP32_EPS = np.finfo(np.float32).eps


def _fro(x):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


# ── metrics ────────────────────────────────────────────────────────────────
def recon_error_svd(A, U, S, Vh) -> float:
    """‖A − U·diag(S)·Vh‖_F / ‖A‖_F  (reduced or full both fine)."""
    A = np.asarray(A, np.float64)
    U, S, Vh = np.asarray(U, np.float64), np.asarray(S, np.float64), np.asarray(Vh, np.float64)
    k = S.shape[0]
    Ar = (U[:, :k] * S) @ Vh[:k, :]
    return _fro(A - Ar) / max(_fro(A), 1e-30)


def recon_error_eigh(A, w, V) -> float:
    """‖A − V·diag(w)·Vᵀ‖_F / ‖A‖_F."""
    A = np.asarray(A, np.float64)
    w, V = np.asarray(w, np.float64), np.asarray(V, np.float64)
    return _fro(A - (V * w) @ V.T) / max(_fro(A), 1e-30)


def orthogonality_error(Q) -> float:
    """‖QᵀQ − I‖_F for the (economy) factor Q."""
    Q = np.asarray(Q, np.float64)
    k = Q.shape[1]
    return _fro(Q.T @ Q - np.eye(k))


def values_rel_error(got, ref) -> float:
    """Max relative error between two sorted value vectors (ascending)."""
    got = np.sort(np.asarray(got, np.float64))
    ref = np.sort(np.asarray(ref, np.float64))
    scale = max(np.max(np.abs(ref)), 1e-30)
    return float(np.max(np.abs(got - ref)) / scale)


# ── pathological test matrices ──────────────────────────────────────────────
def symmetric_cases(n=128, seed=0):
    """Dict of symmetric matrices (float32) that stress an eigensolver."""
    rng = np.random.default_rng(seed)
    out = {}
    A = rng.standard_normal((n, n)).astype("f4"); out["random_sym"] = ((A + A.T) / 2)
    out["identity"] = np.eye(n, dtype="f4")
    out["diagonal"] = np.diag(rng.standard_normal(n).astype("f4"))
    # clustered eigenvalues (hard for vector accuracy)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.concatenate([np.full(n // 2, 1.0), np.full(n - n // 2, 1.0 + 1e-4)]).astype("f4")
    out["clustered_eig"] = (Q * ev) @ Q.T
    # wide spectrum / ill-conditioned
    ev = np.logspace(0, 8, n).astype("f4")
    out["ill_conditioned"] = (Q * ev) @ Q.T
    # rank-deficient
    ev = np.concatenate([rng.standard_normal(n // 2), np.zeros(n - n // 2)]).astype("f4")
    out["rank_deficient"] = (Q * ev) @ Q.T
    # extreme scale
    out["tiny_scale"] = out["random_sym"] * 1e-12
    out["huge_scale"] = out["random_sym"] * 1e12
    return {k: v.astype("f4") for k, v in out.items()}


def svd_cases(m=160, n=96, seed=0):
    """Dict of rectangular/square matrices (float32) for SVD."""
    rng = np.random.default_rng(seed)
    out = {}
    out["tall"] = rng.standard_normal((m, n)).astype("f4")
    out["wide"] = rng.standard_normal((n, m)).astype("f4")
    out["square"] = rng.standard_normal((n, n)).astype("f4")
    # rank-deficient tall
    U = rng.standard_normal((m, n // 2)); V = rng.standard_normal((n // 2, n))
    out["rank_deficient"] = (U @ V).astype("f4")
    # geometric singular values (ill-conditioned)
    Uq, _ = np.linalg.qr(rng.standard_normal((m, n)))
    Vq, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sv = np.logspace(0, 6, n)
    out["ill_conditioned"] = (Uq * sv) @ Vq.T
    return {k: v.astype("f4") for k, v in out.items()}
