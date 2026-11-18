#pragma once
#include "common.cuh"

// Q8_0 x F32 matrix-vector product for GPUs without __dp4a (compute capability < 6.1, e.g. Tesla K80 / sm_37).
// MMVQ emulates dp4a there with byte extracts and integer multiplies and reaches ~40 GB/s on a K80 die; this kernel
// converts the int8 weights to float and uses FMA against the F32 activations, several rows per warp so each x load
// is shared, ~100-150 GB/s. Disable with GGML_CUDA_NO_KEPLER_MMV=1.
bool ggml_cuda_should_use_mmv_kepler(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_vec_kepler(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
