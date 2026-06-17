/**
 * Metal compute kernels for the GPU linalg project (Phase 0 scaffold).
 *
 * Phase 0 ships two kernels whose only job is to PROVE the integration path
 * end-to-end (torch.mps.compile_shader -> zero-copy dispatch on MPS tensors):
 *
 *   1. saxpy             — trivial elementwise; verifies compile + dispatch + I/O
 *   2. apply_col_rotation — applies a Givens/Jacobi rotation to two columns of a
 *                           row-major matrix. This is NOT throwaway: it's the core
 *                           primitive the Phase 1 Jacobi eigensolver/SVD reuses.
 *
 * Storage convention: all matrices are row-major float32, element (i, j) at i*n + j.
 */

#include <metal_stdlib>
using namespace metal;

// out[i] = a * x[i] + y[i]
kernel void saxpy(
    device const float* x   [[buffer(0)]],
    device const float* y   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant float&     a   [[buffer(3)]],
    constant uint&      count [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= count) return;
    out[i] = a * x[i] + y[i];
}

// Right-multiply A (n x n, row-major) by a Givens rotation in the (p, q) plane:
//   col_p' =  c*col_p - s*col_q
//   col_q' =  s*col_p + c*col_q
// One thread per row i. In-place on A. This is the column-rotation building
// block of one-sided Jacobi SVD (and, applied to rows too, two-sided Jacobi eigh).
kernel void apply_col_rotation(
    device float*  A [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant uint& p [[buffer(2)]],
    constant uint& q [[buffer(3)]],
    constant float& c [[buffer(4)]],
    constant float& s [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    const uint ip = i * n + p;
    const uint iq = i * n + q;
    const float ap = A[ip];
    const float aq = A[iq];
    A[ip] = c * ap - s * aq;
    A[iq] = s * ap + c * aq;
}

// ─── Phase 1: two-sided cyclic Jacobi symmetric eigendecomposition ──────────
//
// One threadgroup of TG threads cooperatively diagonalizes a symmetric n x n
// matrix A (row-major, in device memory) by a sweep of Jacobi rotations, while
// accumulating the eigenvectors into V (pre-initialized to the identity).
//
//   A  -> destroyed; its diagonal holds the eigenvalues on return
//   V  -> identity in, eigenvectors (columns) out
//
// Correctness-first: A and V live in device memory (no threadgroup-memory size
// limit, so any n works) and a single threadgroup does the work. This is slow at
// small n (CPU/AMX wins there) — the point of Phase 1 is to prove the algorithm
// and the kernel mechanics. Phase 2 adds tiling / multiple threadgroups for speed.
//
// TG must be a power of two and equal the dispatch's threads-per-threadgroup.
constant uint TG = 256;

// Sum of squares of the off-diagonal entries (counts both triangles), tree-reduced.
inline float off_diag_norm2(device const float* A, uint n, uint tid, uint tcount,
                            threadgroup float* tg)
{
    float local = 0.0f;
    for (uint idx = tid; idx < n * n; idx += tcount) {
        if (idx / n != idx % n) { float a = A[idx]; local += a * a; }
    }
    tg[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) tg[tid] += tg[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return tg[0];
}

kernel void jacobi_eigh(
    device float*  A          [[buffer(0)]],
    device float*  V          [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    constant uint& max_sweeps [[buffer(3)]],
    constant float& tol       [[buffer(4)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    threadgroup float tg[TG];

    const float init_off2 = off_diag_norm2(A, n, tid, tcount, tg);
    const float thresh = tol * tol * init_off2;   // converged when off² ≤ thresh

    for (uint sweep = 0; sweep < max_sweeps; ++sweep) {
        for (uint p = 0; p + 1 < n; ++p) {
            for (uint q = p + 1; q < n; ++q) {
                // every thread computes the same rotation from the same globals
                const float apq = A[p * n + q];
                float c = 1.0f, s = 0.0f;
                if (fabs(apq) > 1e-30f) {
                    const float app = A[p * n + p];
                    const float aqq = A[q * n + q];
                    const float tau = (aqq - app) / (2.0f * apq);
                    const float t = (tau >= 0.0f ? 1.0f : -1.0f) /
                                    (fabs(tau) + sqrt(1.0f + tau * tau));
                    c = rsqrt(1.0f + t * t);
                    s = t * c;
                }
                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

                if (s != 0.0f) {
                    // A <- (A J): rotate columns p,q  (A[k,p], A[k,q] for all rows k)
                    for (uint k = tid; k < n; k += tcount) {
                        const float akp = A[k * n + p];
                        const float akq = A[k * n + q];
                        A[k * n + p] = c * akp - s * akq;
                        A[k * n + q] = s * akp + c * akq;
                    }
                    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                    // A <- (Jᵀ A): rotate rows p,q using the column-updated matrix
                    for (uint k = tid; k < n; k += tcount) {
                        const float apk = A[p * n + k];
                        const float aqk = A[q * n + k];
                        A[p * n + k] = c * apk - s * aqk;
                        A[q * n + k] = s * apk + c * aqk;
                    }
                    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                    // V <- V J: accumulate eigenvectors (column rotation)
                    for (uint k = tid; k < n; k += tcount) {
                        const float vkp = V[k * n + p];
                        const float vkq = V[k * n + q];
                        V[k * n + p] = c * vkp - s * vkq;
                        V[k * n + q] = s * vkp + c * vkq;
                    }
                    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                }
            }
        }
        if (off_diag_norm2(A, n, tid, tcount, tg) <= thresh) break;
    }
}

// ─── Phase 2: BATCHED Jacobi eigh — the actual GPU-speedup path ─────────────
//
// One threadgroup per matrix; the whole grid runs B matrices concurrently across
// the GPU's cores. Each matrix (n <= MAX_BN) and its eigenvector accumulator live
// in THREADGROUP MEMORY, so all Jacobi sweeps run with zero global-memory traffic
// (one load in, one store out). This is where the GPU beats the CPU: many small
// independent eigendecompositions, fully parallel.
//
//   A : [B, n, n] row-major; per-matrix block destroyed, diagonal = eigenvalues
//   V : [B, n, n]; eigenvectors out (initialized to identity inside the kernel)
//
// BATCH_MAX_BN bounds the threadgroup-memory footprint: sA + sV = 2*MAX_BN^2
// floats. Smaller footprint + fewer threads/matrix (BATCH_BTG) => more matrices
// resident per core => higher throughput. Both are #defines so the Python side
// compiles a specialization tuned to the actual n (occupancy autotuning).
#ifndef BATCH_MAX_BN
#define BATCH_MAX_BN 32
#endif
#ifndef BATCH_BTG
#define BATCH_BTG 64               // threads per matrix (power of two)
#endif
// Storage precision for the on-chip matrix. Default float (fp32); define =half for
// the fp16 variant — halves threadgroup memory (higher occupancy). Compute stays
// fp32 (locals are float; MSL converts on load/store), so only storage is reduced.
#ifndef BATCH_STORE
#define BATCH_STORE float
#endif

kernel void batched_jacobi_eigh(
    device float*  A          [[buffer(0)]],
    device float*  V          [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    constant uint& max_sweeps [[buffer(3)]],
    constant float& tol       [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint b   [[threadgroup_position_in_grid]])
{
    threadgroup BATCH_STORE sA[BATCH_MAX_BN * BATCH_MAX_BN];
    threadgroup BATCH_STORE sV[BATCH_MAX_BN * BATCH_MAX_BN];
    threadgroup float tg[BATCH_BTG];

    const uint nn = n * n;
    device float* Ab = A + b * nn;
    device float* Vb = V + b * nn;

    // load this matrix into threadgroup memory; sV = identity
    for (uint idx = tid; idx < nn; idx += BATCH_BTG) {
        sA[idx] = Ab[idx];
        sV[idx] = (idx / n == idx % n) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // initial off-diagonal norm (threadgroup-memory reduction)
    auto off2 = [&]() -> float {
        float local = 0.0f;
        for (uint idx = tid; idx < nn; idx += BATCH_BTG)
            if (idx / n != idx % n) { float a = sA[idx]; local += a * a; }
        tg[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = BATCH_BTG / 2; s > 0; s >>= 1) {
            if (tid < s) tg[tid] += tg[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return tg[0];
    };
    const float thresh = tol * tol * off2();

    for (uint sweep = 0; sweep < max_sweeps; ++sweep) {
        for (uint p = 0; p + 1 < n; ++p) {
            for (uint q = p + 1; q < n; ++q) {
                const float apq = sA[p * n + q];
                float c = 1.0f, s = 0.0f;
                if (fabs(apq) > 1e-30f) {
                    const float app = sA[p * n + p];
                    const float aqq = sA[q * n + q];
                    const float tau = (aqq - app) / (2.0f * apq);
                    const float t = (tau >= 0.0f ? 1.0f : -1.0f) /
                                    (fabs(tau) + sqrt(1.0f + tau * tau));
                    c = rsqrt(1.0f + t * t);
                    s = t * c;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (s != 0.0f) {
                    for (uint k = tid; k < n; k += BATCH_BTG) {   // columns p,q
                        const float akp = sA[k * n + p];
                        const float akq = sA[k * n + q];
                        sA[k * n + p] = c * akp - s * akq;
                        sA[k * n + q] = s * akp + c * akq;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint k = tid; k < n; k += BATCH_BTG) {   // rows p,q
                        const float apk = sA[p * n + k];
                        const float aqk = sA[q * n + k];
                        sA[p * n + k] = c * apk - s * aqk;
                        sA[q * n + k] = s * apk + c * aqk;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint k = tid; k < n; k += BATCH_BTG) {   // V <- V J
                        const float vkp = sV[k * n + p];
                        const float vkq = sV[k * n + q];
                        sV[k * n + p] = c * vkp - s * vkq;
                        sV[k * n + q] = s * vkp + c * vkq;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
        if (off2() <= thresh) break;
    }

    for (uint idx = tid; idx < nn; idx += BATCH_BTG) {   // store back
        Ab[idx] = sA[idx];
        Vb[idx] = sV[idx];
    }
}

// ─── Phase 3: BATCHED one-sided (Hestenes) Jacobi SVD ──────────────────────
//
// One threadgroup per matrix; columns of A (m x n, tall: m >= n) are orthogonalized
// in threadgroup memory by right-side Jacobi rotations, while V accumulates the
// rotations. At convergence: column norms are the singular values, the normalized
// columns are U, and V holds the right singular vectors. Threadgroup-resident, so
// the throughput win mirrors batched eigh; one-sided Jacobi is accurate for small
// singular values (unlike a Gram/AᵀA approach).
//
//   A : [B, m, n]  -> overwritten with U (economy, m x n, orthonormal columns)
//   V : [B, n, n]  -> right singular vectors (V, not Vᵀ)
//   S : [B, n]     -> singular values (unsorted; the host sorts descending)
//
// Convergence is detected with zero communication: every thread computes the same
// column dot products (from the shared reduction) and the same rotate/skip decision,
// so a per-thread register flag agrees across the threadgroup.
#ifndef SVD_MAXM
#define SVD_MAXM 64
#endif
#ifndef SVD_MAXN
#define SVD_MAXN 32
#endif
#ifndef SVD_BTG
#define SVD_BTG 32
#endif

kernel void batched_jacobi_svd(
    device float*  A          [[buffer(0)]],
    device float*  V          [[buffer(1)]],
    device float*  S          [[buffer(2)]],
    constant uint& m          [[buffer(3)]],
    constant uint& n          [[buffer(4)]],
    constant uint& max_sweeps [[buffer(5)]],
    constant float& tol       [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint b   [[threadgroup_position_in_grid]])
{
    threadgroup float sA[SVD_MAXM * SVD_MAXN];
    threadgroup float sV[SVD_MAXN * SVD_MAXN];
    threadgroup float ta[SVD_BTG], tb[SVD_BTG], tc[SVD_BTG];

    device float* Ab = A + b * m * n;
    device float* Vb = V + b * n * n;
    device float* Sb = S + b * n;

    for (uint idx = tid; idx < m * n; idx += SVD_BTG) sA[idx] = Ab[idx];
    for (uint idx = tid; idx < n * n; idx += SVD_BTG)
        sV[idx] = (idx / n == idx % n) ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint sweep = 0; sweep < max_sweeps; ++sweep) {
        bool any_rot = false;
        for (uint i = 0; i + 1 < n; ++i) {
            for (uint j = i + 1; j < n; ++j) {
                // column dot products over the m rows (combined 3-way reduction)
                float la = 0.0f, lb = 0.0f, lg = 0.0f;
                for (uint k = tid; k < m; k += SVD_BTG) {
                    const float ai = sA[k * n + i], aj = sA[k * n + j];
                    la += ai * ai; lb += aj * aj; lg += ai * aj;
                }
                ta[tid] = la; tb[tid] = lb; tc[tid] = lg;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = SVD_BTG / 2; s > 0; s >>= 1) {
                    if (tid < s) { ta[tid] += ta[tid + s]; tb[tid] += tb[tid + s]; tc[tid] += tc[tid + s]; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                const float alpha = ta[0], beta = tb[0], gamma = tc[0];

                if (alpha > 0.0f && beta > 0.0f &&
                    fabs(gamma) > tol * sqrt(alpha * beta)) {
                    any_rot = true;
                    const float zeta = (beta - alpha) / (2.0f * gamma);
                    const float t = (zeta >= 0.0f ? 1.0f : -1.0f) /
                                    (fabs(zeta) + sqrt(1.0f + zeta * zeta));
                    const float c = rsqrt(1.0f + t * t), s = t * c;
                    for (uint k = tid; k < m; k += SVD_BTG) {   // rotate cols i,j of A
                        const float ai = sA[k * n + i], aj = sA[k * n + j];
                        sA[k * n + i] = c * ai - s * aj;
                        sA[k * n + j] = s * ai + c * aj;
                    }
                    for (uint k = tid; k < n; k += SVD_BTG) {   // rotate cols i,j of V
                        const float vi = sV[k * n + i], vj = sV[k * n + j];
                        sV[k * n + i] = c * vi - s * vj;
                        sV[k * n + j] = s * vi + c * vj;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
        if (!any_rot) break;
    }

    // singular values = column norms of the orthogonalized A
    for (uint i = tid; i < n; i += SVD_BTG) {
        float nrm = 0.0f;
        for (uint k = 0; k < m; ++k) { const float a = sA[k * n + i]; nrm += a * a; }
        Sb[i] = sqrt(nrm);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // U = normalized columns of A; write U and V back
    for (uint idx = tid; idx < m * n; idx += SVD_BTG) {
        const float nrm = Sb[idx % n];
        Ab[idx] = (nrm > 1e-30f) ? sA[idx] / nrm : 0.0f;
    }
    for (uint idx = tid; idx < n * n; idx += SVD_BTG) Vb[idx] = sV[idx];
}

// ─── Phase 2b: BATCHED eigh for larger n (32 < n <= 64), V in global memory ──
//
// For n up to 64, A (n*n floats) still fits in threadgroup memory but A+V (2*n*n)
// would not, so only A is threadgroup-resident; the eigenvector accumulator V
// lives in device memory (the V rotation is the only loop that touches global).
// Widens the batched regime past the n<=32 fully-resident path.
#ifndef BATCH_BIG_MAXN
#define BATCH_BIG_MAXN 64
#endif
#ifndef BATCH_BIG_BTG
#define BATCH_BIG_BTG 64
#endif

kernel void batched_jacobi_eigh_vg(
    device float*  A          [[buffer(0)]],
    device float*  V          [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    constant uint& max_sweeps [[buffer(3)]],
    constant float& tol       [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint b   [[threadgroup_position_in_grid]])
{
    threadgroup float sA[BATCH_BIG_MAXN * BATCH_BIG_MAXN];
    threadgroup float tg[BATCH_BIG_BTG];

    const uint nn = n * n;
    device float* Ab = A + b * nn;
    device float* Vb = V + b * nn;

    for (uint idx = tid; idx < nn; idx += BATCH_BIG_BTG) {
        sA[idx] = Ab[idx];
        Vb[idx] = (idx / n == idx % n) ? 1.0f : 0.0f;   // V = identity (device)
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    auto off2 = [&]() -> float {
        float local = 0.0f;
        for (uint idx = tid; idx < nn; idx += BATCH_BIG_BTG)
            if (idx / n != idx % n) { float a = sA[idx]; local += a * a; }
        tg[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = BATCH_BIG_BTG / 2; s > 0; s >>= 1) {
            if (tid < s) tg[tid] += tg[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return tg[0];
    };
    const float thresh = tol * tol * off2();

    for (uint sweep = 0; sweep < max_sweeps; ++sweep) {
        for (uint p = 0; p + 1 < n; ++p) {
            for (uint q = p + 1; q < n; ++q) {
                const float apq = sA[p * n + q];
                float c = 1.0f, s = 0.0f;
                if (fabs(apq) > 1e-30f) {
                    const float app = sA[p * n + p], aqq = sA[q * n + q];
                    const float tau = (aqq - app) / (2.0f * apq);
                    const float t = (tau >= 0.0f ? 1.0f : -1.0f) /
                                    (fabs(tau) + sqrt(1.0f + tau * tau));
                    c = rsqrt(1.0f + t * t); s = t * c;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (s != 0.0f) {
                    for (uint k = tid; k < n; k += BATCH_BIG_BTG) {   // cols p,q (sA)
                        const float akp = sA[k * n + p], akq = sA[k * n + q];
                        sA[k * n + p] = c * akp - s * akq;
                        sA[k * n + q] = s * akp + c * akq;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint k = tid; k < n; k += BATCH_BIG_BTG) {   // rows p,q (sA)
                        const float apk = sA[p * n + k], aqk = sA[q * n + k];
                        sA[p * n + k] = c * apk - s * aqk;
                        sA[q * n + k] = s * apk + c * aqk;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint k = tid; k < n; k += BATCH_BIG_BTG) {   // V <- V J (device)
                        const float vkp = Vb[k * n + p], vkq = Vb[k * n + q];
                        Vb[k * n + p] = c * vkp - s * vkq;
                        Vb[k * n + q] = s * vkp + c * vkq;
                    }
                    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                }
            }
        }
        if (off2() <= thresh) break;
    }
    for (uint idx = tid; idx < nn; idx += BATCH_BIG_BTG) Ab[idx] = sA[idx];  // diagonal=eigvals
}
