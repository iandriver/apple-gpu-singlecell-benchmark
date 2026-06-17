"""
Phase 7: the complete torch.linalg-on-MPS shim.

Verifies that, after install(), the whole factorization/solver surface works on MPS
tensors — correct vs CPU LAPACK, returning MPS tensors with the right return type —
across accelerated ops (eigh/eigvalsh/svd/svdvals) and CPU-fallback ops (qr/pinv/
lstsq/eigvals/slogdet). Also: batched eigh/svd still run on the GPU; CPU inputs are
untouched; uninstall() restores everything.

Run:  python -m metal_linalg.test_phase7
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import metal_linalg

_PASS, _FAIL = 0, 0


def check(name, ok, detail=""):
    global _PASS, _FAIL
    _PASS += ok
    _FAIL += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _close(a, b, tol=1e-3):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b))) < tol


def main():
    print(f"torch {torch.__version__} | Phase 7 full torch.linalg-on-MPS shim\n")
    rng = np.random.default_rng(0)
    metal_linalg.install()

    print("1) Accelerated ops on MPS (batched-small -> GPU kernels)")
    S = rng.standard_normal((64, 20, 20)).astype("f4"); S = (S + S.transpose(0, 2, 1)) / 2
    Sm = torch.from_numpy(S).to("mps")
    w, V = torch.linalg.eigh(Sm)
    check("eigh: mps out + correct", w.device.type == "mps"
          and _close(w[0].cpu(), np.linalg.eigvalsh(S[0])))
    wv = torch.linalg.eigvalsh(Sm)
    check("eigvalsh: mps out + correct", wv.device.type == "mps"
          and _close(np.sort(wv[0].cpu()), np.sort(np.linalg.eigvalsh(S[0]))))
    R = rng.standard_normal((64, 40, 16)).astype("f4")
    Rm = torch.from_numpy(R).to("mps")
    U, sv, Vh = torch.linalg.svd(Rm, full_matrices=False)
    check("svd: mps out + correct", U.device.type == "mps"
          and _close(np.sort(sv[0].cpu()), np.sort(np.linalg.svd(R[0], compute_uv=False))))
    svv = torch.linalg.svdvals(Rm)
    check("svdvals: mps out + correct", svv.device.type == "mps"
          and _close(np.sort(svv[0].cpu()), np.sort(np.linalg.svd(R[0], compute_uv=False))))

    print("\n2) CPU-fallback ops on MPS (no GPU kernel -> transparent round-trip)")
    M = torch.from_numpy(rng.standard_normal((50, 30)).astype("f4")).to("mps")
    Q, Rqr = torch.linalg.qr(M)            # qr hangs natively on MPS; shim forces CPU
    check("qr: mps out + reconstructs", Q.device.type == "mps"
          and _close((Q @ Rqr).cpu(), M.cpu()))
    P = torch.linalg.pinv(M)
    check("pinv: mps out + A P A == A", P.device.type == "mps"
          and _close((M @ P @ M).cpu(), M.cpu(), tol=1e-2))
    A = torch.from_numpy(rng.standard_normal((40, 20)).astype("f4")).to("mps")
    b = torch.from_numpy(rng.standard_normal((40, 1)).astype("f4")).to("mps")
    sol = torch.linalg.lstsq(A, b).solution
    check("lstsq: mps out + matches numpy", sol.device.type == "mps"
          and _close(sol.cpu().ravel(), np.linalg.lstsq(A.cpu().numpy(),
                     b.cpu().numpy(), rcond=None)[0].ravel(), tol=1e-2))
    sq = torch.from_numpy(rng.standard_normal((10, 10)).astype("f4")).to("mps")
    ev = torch.linalg.eigvals(sq)
    check("eigvals: mps out", ev.device.type == "mps")
    sgn, logabs = torch.linalg.slogdet(sq)
    check("slogdet: mps out + matches numpy", logabs.device.type == "mps"
          and _close(logabs.cpu(), np.linalg.slogdet(sq.cpu().numpy())[1], tol=1e-2))

    print("\n3) Large single matrix (out of batched range -> CPU round-trip, still MPS out)")
    big = torch.from_numpy((lambda X: (X + X.T) / 2)(
        rng.standard_normal((300, 300)).astype("f4"))).to("mps")
    wb, _ = torch.linalg.eigh(big)
    check("eigh(300x300 mps) works + mps out", wb.device.type == "mps")

    print("\n4) CPU inputs untouched (original LAPACK)")
    Ac = torch.from_numpy(S[0])
    wc = torch.linalg.eigvalsh(Ac)
    check("cpu path unchanged + correct", wc.device.type == "cpu"
          and _close(wc.numpy(), np.linalg.eigvalsh(S[0])))

    print("\n5) uninstall restores originals")
    metal_linalg.uninstall()
    try:
        torch.linalg.eigh(Sm)
        check("eigh(mps) raises again after uninstall", False)
    except Exception:
        check("eigh(mps) raises again after uninstall", True)

    print(f"\n{'=' * 48}\n{_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
