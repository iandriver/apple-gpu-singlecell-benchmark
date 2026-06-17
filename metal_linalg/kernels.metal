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

kernel void batched_jacobi_eigh(
    device float*  A          [[buffer(0)]],
    device float*  V          [[buffer(1)]],
    constant uint& n          [[buffer(2)]],
    constant uint& max_sweeps [[buffer(3)]],
    constant float& tol       [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint b   [[threadgroup_position_in_grid]])
{
    threadgroup float sA[BATCH_MAX_BN * BATCH_MAX_BN];
    threadgroup float sV[BATCH_MAX_BN * BATCH_MAX_BN];
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
