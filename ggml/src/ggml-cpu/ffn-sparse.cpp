// Per-token sparse FFN on the CPU (wllm fork, 2026-11-27).
//
// Two ggml_custom_op_t callbacks, reached through the CPU backend's get_proc_address ("ggml_cpu_ffn_sparse_gateup",
// "ggml_cpu_ffn_sparse_down") so that llama can build them with ggml_custom_4d also when the backends are loaded as
// plugins (GGML_BACKEND_DL). A predictor picks k of the n_ff units per token; only those rows of ffn_gate / ffn_up
// are multiplied, and only those rows of the transposed ffn_down (one contiguous row of n_embd per unit) are summed.
//
//   gateup: dst F32 [k, n_tok];      src0 x F32 [n_embd, n_tok], src1 gate [n_embd, n_ff], src2 up [n_embd, n_ff],
//           src3 ids I32 [k, n_tok]. dst[j, t] = silu(gate[ids[j, t]] . x_t) * (up[ids[j, t]] . x_t); relu instead of
//           silu when userdata is non-null.
//           Threads split the k units.
//   down:   dst F32 [n_embd, n_tok]; src0 a F32 [k, n_tok], src1 down_t [n_embd, n_ff], src2 ids I32 [k, n_tok].
//           dst[:, t] = sum_j a[j, t] down_t[ids[j, t]]. Threads split the n_embd outputs in whole quant blocks, so
//           no reduction between threads is needed.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct q8_0_block {      // layout of block_q8_0 (ggml-common.h)
    ggml_fp16_t d;
    int8_t      qs[32];
};
static_assert(sizeof(q8_0_block) == 34, "block_q8_0 layout");

const void * quantized_row(const ggml_type_traits_cpu * tt, const float * x, int64_t n, std::vector<uint8_t> & buf) {
    if (tt->vec_dot_type == GGML_TYPE_F32) {
        return x;
    }
    buf.resize(ggml_row_size(tt->vec_dot_type, n));
    ggml_get_type_traits_cpu(tt->vec_dot_type)->from_float(x, buf.data(), n);
    return buf.data();
}

} // namespace

extern "C" {

void ggml_cpu_ffn_sparse_gateup(struct ggml_tensor * dst, int ith, int nth, void * userdata);
void ggml_cpu_ffn_sparse_down(struct ggml_tensor * dst, int ith, int nth, void * userdata);

void ggml_cpu_ffn_sparse_gateup(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const bool relu = userdata != nullptr;   // non-null: gated ReLU (ReLU-fied models), else SwiGLU
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * wg  = dst->src[1];
    const ggml_tensor * wu  = dst->src[2];
    const ggml_tensor * ids = dst->src[3];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    const int64_t n_embd = x->ne[0];
    const int64_t n_tok  = x->ne[1];
    const int64_t k      = ids->ne[0];
    GGML_ASSERT(wg->ne[0] == n_embd && wu->ne[0] == n_embd && dst->ne[0] == k && dst->ne[1] == n_tok);

    const ggml_type_traits_cpu * tg = ggml_get_type_traits_cpu(wg->type);
    const ggml_type_traits_cpu * tu = ggml_get_type_traits_cpu(wu->type);
    const int64_t j0 = k * ith / nth;
    const int64_t j1 = k * (ith + 1) / nth;
    if (j0 >= j1) {
        return;
    }
    std::vector<uint8_t> bg, bu;
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * xr = (const float *) ((const char *) x->data + t * x->nb[1]);
        const void  * qg = quantized_row(tg, xr, n_embd, bg);
        const void  * qu = tu->vec_dot_type == tg->vec_dot_type ? qg : quantized_row(tu, xr, n_embd, bu);
        const int32_t * id = (const int32_t *) ((const char *) ids->data + t * ids->nb[1]);
        float * out = (float *) ((char *) dst->data + t * dst->nb[1]);
        for (int64_t j = j0; j < j1; ++j) {
            const int64_t u = id[j];
            float g = 0.0f, v = 0.0f;
            tg->vec_dot((int) n_embd, &g, 0, (const char *) wg->data + u * wg->nb[1], 0, qg, 0, 1);
            tu->vec_dot((int) n_embd, &v, 0, (const char *) wu->data + u * wu->nb[1], 0, qu, 0, 1);
            out[j] = relu ? (g > 0.0f ? g * v : 0.0f) : g / (1.0f + expf(-g)) * v;
        }
    }
}

void ggml_cpu_ffn_sparse_down(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;
    const ggml_tensor * a   = dst->src[0];
    const ggml_tensor * w   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    GGML_ASSERT(a->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    const int64_t n_embd = w->ne[0];
    const int64_t n_tok  = a->ne[1];
    const int64_t k      = a->ne[0];
    GGML_ASSERT(dst->ne[0] == n_embd && dst->ne[1] == n_tok && ids->ne[0] == k);

    const int64_t blck = ggml_blck_size(w->type);
    const int64_t nblk = n_embd / blck;
    const int64_t b0 = nblk * ith / nth;
    const int64_t b1 = nblk * (ith + 1) / nth;
    if (b0 >= b1) {
        return;
    }
    const int64_t c0  = b0 * blck;
    const int64_t nc  = (b1 - b0) * blck;
    const size_t  off = (size_t) b0 * ggml_type_size(w->type);
    std::vector<float> acc(nc), tmp;
    const ggml_to_float_t to_float = ggml_get_type_traits(w->type)->to_float;
    GGML_ASSERT(w->type == GGML_TYPE_Q8_0 || w->type == GGML_TYPE_F32 || to_float);

    for (int64_t t = 0; t < n_tok; ++t) {
        std::memset(acc.data(), 0, nc * sizeof(float));
        const float   * ar = (const float *) ((const char *) a->data + t * a->nb[1]);
        const int32_t * id = (const int32_t *) ((const char *) ids->data + t * ids->nb[1]);
        for (int64_t j = 0; j < k; ++j) {
            const float av = ar[j];
            if (av == 0.0f) {
                continue;
            }
            const char * row = (const char *) w->data + (int64_t) id[j] * w->nb[1] + off;
            if (w->type == GGML_TYPE_Q8_0) {
                const q8_0_block * bl = (const q8_0_block *) row;
                for (int64_t b = 0; b < b1 - b0; ++b) {
                    const float s = av * ggml_fp16_to_fp32(bl[b].d);
                    float * o = acc.data() + b * 32;
                    for (int i = 0; i < 32; ++i) {
                        o[i] += s * bl[b].qs[i];
                    }
                }
            } else if (w->type == GGML_TYPE_F32) {
                const float * r = (const float *) row;
                for (int64_t c = 0; c < nc; ++c) {
                    acc[c] += av * r[c];
                }
            } else {
                tmp.resize(nc);
                to_float(row, tmp.data(), nc);
                for (int64_t c = 0; c < nc; ++c) {
                    acc[c] += av * tmp[c];
                }
            }
        }
        std::memcpy((char *) dst->data + t * dst->nb[1] + c0 * sizeof(float), acc.data(), nc * sizeof(float));
    }
}

} // extern "C"
