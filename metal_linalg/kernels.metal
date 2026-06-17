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
