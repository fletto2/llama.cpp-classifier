#include "mmv-kepler.cuh"

#include <cstdlib>

// block_q8_0 is 34 bytes (fp16 scale + 32 int8), so its quants are only 2-byte aligned. Block b of a row starts at byte
// 34*b: 4-byte aligned for even b, 2 past it for odd b. Read the block as 9 aligned 32-bit words and realign with
// __byte_perm. Rows must start 4-byte aligned: ne00 % 64 == 0 (an even number of blocks per row).
template <int R>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_kepler(
        const char * __restrict__ vx, const float * __restrict__ x, float * __restrict__ y,
        const int nrows, const int nblocks, const int64_t row_stride) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * (blockDim.x / WARP_SIZE) + warp) * R;
    if (row0 >= nrows) {
        return;
    }

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }

    for (int b = lane; b < nblocks; b += WARP_SIZE) {
        const float4 * xb = reinterpret_cast<const float4 *>(x + b * QK8_0);
        float4 xv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            xv[j] = __ldg(xb + j);
        }
        const int64_t off = (int64_t) b * sizeof(block_q8_0);
        const bool    odd = off & 3;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= nrows) {
                break;
            }
            const uint32_t * p = reinterpret_cast<const uint32_t *>(vx + row * row_stride + (off & ~(int64_t) 3));
            uint32_t u[9];
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                u[i] = __ldg(p + i);
            }
            const float d = __half2float(__ushort_as_half((unsigned short) (odd ? (u[0] >> 16) : (u[0] & 0xFFFF))));
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t v = odd ? u[j + 1] : __byte_perm(u[j], u[j + 1], 0x5432);
                s = fmaf((float) (int8_t) (v      ), xv[j].x, s);
                s = fmaf((float) (int8_t) (v >>  8), xv[j].y, s);
                s = fmaf((float) (int8_t) (v >> 16), xv[j].z, s);
                s = fmaf((float) (int8_t) (v >> 24), xv[j].w, s);
            }
            acc[r] = fmaf(d, s, acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_reduce_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) {
            y[row0 + r] = v;
        }
    }
}

// block_q4_0 is 18 bytes (fp16 scale + 16 bytes of nibble pairs): byte j holds element j (low nibble) and j + 16
// (high nibble), value (nibble - 8) * d. Block b starts at 18*b: 4-byte aligned for even b, 2 past it for odd b.
// 5 aligned words cover it. Rows start 4-byte aligned when ne00 % 64 == 0.
template <int R>
static __global__ void __launch_bounds__(256) mul_mat_vec_q4_0_kepler(
        const char * __restrict__ vx, const float * __restrict__ x, float * __restrict__ y,
        const int nrows, const int nblocks, const int64_t row_stride) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x * (blockDim.x / WARP_SIZE) + warp) * R;
    if (row0 >= nrows) {
        return;
    }

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.0f;
    }

    for (int b = lane; b < nblocks; b += WARP_SIZE) {
        const float4 * xb = reinterpret_cast<const float4 *>(x + b * QK4_0);
        float4 xv[8];                                   // xv[0..3]: elements 0..15, xv[4..7]: 16..31
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            xv[j] = __ldg(xb + j);
        }
        // sum of x over the block, for the -8 offset: sum((n - 8) x) = sum(n x) - 8 sum(x)
        float xs = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            xs += (xv[j].x + xv[j].y) + (xv[j].z + xv[j].w);
        }
        const int64_t off = (int64_t) b * sizeof(block_q4_0);
        const bool    odd = off & 3;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= nrows) {
                break;
            }
            const uint32_t * p = reinterpret_cast<const uint32_t *>(vx + row * row_stride + (off & ~(int64_t) 3));
            uint32_t u[5];
#pragma unroll
            for (int i = 0; i < 5; ++i) {
                u[i] = __ldg(p + i);
            }
            const float d = __half2float(__ushort_as_half((unsigned short) (odd ? (u[0] >> 16) : (u[0] & 0xFFFF))));
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {               // quant word j = bytes 4j..4j+3 = elements 4j..4j+3 (+16 high)
                const uint32_t v = odd ? u[j + 1] : __byte_perm(u[j], u[j + 1], 0x5432);
                s = fmaf((float) ((v      ) & 0xF), xv[j].x, s);
                s = fmaf((float) ((v >>  8) & 0xF), xv[j].y, s);
                s = fmaf((float) ((v >> 16) & 0xF), xv[j].z, s);
                s = fmaf((float) ((v >> 24) & 0xF), xv[j].w, s);
                s = fmaf((float) ((v >>  4) & 0xF), xv[j + 4].x, s);
                s = fmaf((float) ((v >> 12) & 0xF), xv[j + 4].y, s);
                s = fmaf((float) ((v >> 20) & 0xF), xv[j + 4].z, s);
                s = fmaf((float) ((v >> 28)      ), xv[j + 4].w, s);
            }
            acc[r] = fmaf(d, s - 8.0f * xs, acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = warp_reduce_sum(acc[r]);
        if (lane == 0 && row0 + r < nrows) {
            y[row0 + r] = v;
        }
    }
}

bool ggml_cuda_should_use_mmv_kepler(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    static const bool disabled = getenv("GGML_CUDA_NO_KEPLER_MMV") != nullptr;
    if (disabled || GGML_CUDA_CC_IS_AMD(cc) || GGML_CUDA_CC_IS_MTHREADS(cc) || cc >= GGML_CUDA_CC_DP4A) {
        return false;
    }
    return (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] == 1 && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src0->ne[0] % 64 == 0 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
           (uintptr_t) src0->data % 4 == 0 && (uintptr_t) src1->data % 16 == 0;
}

void ggml_cuda_mul_mat_vec_kepler(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int     nrows   = src0->ne[1];
    const int     nblocks = src0->ne[0] / QK8_0;          // QK4_0 == QK8_0 == 32
    const int64_t stride  = src0->nb[1];
    const char  * vx = (const char  *) src0->data;
    const float * x  = (const float *) src1->data;
    float       * y  = (float       *) dst->data;
    cudaStream_t stream = ctx.stream();
    const dim3 block(8 * WARP_SIZE);
    // more rows per warp when x is too large for the read-only cache (measured on a K80 die, P0)
    if (src0->type == GGML_TYPE_Q4_0) {
        if (src0->ne[0] >= 8192) {
            mul_mat_vec_q4_0_kepler<8><<<(nrows + 63) / 64, block, 0, stream>>>(vx, x, y, nrows, nblocks, stride);
        } else {
            mul_mat_vec_q4_0_kepler<4><<<(nrows + 31) / 32, block, 0, stream>>>(vx, x, y, nrows, nblocks, stride);
        }
    } else if (src0->ne[0] >= 8192) {
        mul_mat_vec_q8_0_kepler<8><<<(nrows + 63) / 64, block, 0, stream>>>(vx, x, y, nrows, nblocks, stride);
    } else {
        mul_mat_vec_q8_0_kepler<4><<<(nrows + 31) / 32, block, 0, stream>>>(vx, x, y, nrows, nblocks, stride);
    }
    CUDA_CHECK(cudaGetLastError());
}
