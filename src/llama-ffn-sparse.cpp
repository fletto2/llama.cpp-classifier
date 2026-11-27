#include "llama-ffn-sparse.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cstring>
#include <string>

int32_t llama_ffn_sparse::load(const llama_model & model, const char * path, float keep_, int32_t first_layer, int32_t max_tokens_) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        LLAMA_LOG_ERROR("%s: no CPU device\n", __func__);
        return -1;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu);
    op_gateup = (ggml_custom_op_t) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_ffn_sparse_gateup");
    op_down   = (ggml_custom_op_t) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_ffn_sparse_down");
    if (!op_gateup || !op_down) {
        LLAMA_LOG_ERROR("%s: the CPU backend has no ffn-sparse kernels\n", __func__);
        return -1;
    }
    ggml_context * src = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &src };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) {
        LLAMA_LOG_ERROR("%s: failed to read %s\n", __func__, path);
        return -1;
    }
    const int n_layer = (int) model.layers.size();
    const int64_t n_embd = model.hparams.n_embd;
    layers.assign(n_layer, layer());
    ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead() * 5 * (size_t) n_layer, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ctx.reset(ggml_init(ip));
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copies;
    int32_t n = 0;
    bool ok = true;
    for (int il = std::max(0, first_layer); il < n_layer && ok; ++il) {
        auto get = [&](const char * what) {
            return ggml_get_tensor(src, (std::string("fsp.") + what + "." + std::to_string(il)).c_str());
        };
        ggml_tensor * t[5] = { get("w1"), get("b1"), get("w2"), get("b2"), get("down_t") };
        if (!t[0]) {
            continue;
        }
        const llama_layer & L = model.layers[il];
        const int64_t n_ff = L.ffn_gate ? L.ffn_gate->ne[1] : 0;
        for (int i = 0; i < 5; ++i) {
            ok = ok && t[i];
        }
        ok = ok && t[0]->ne[0] == n_embd && t[2]->ne[1] == n_ff && t[3]->ne[0] == n_ff && t[2]->ne[0] == t[0]->ne[1]
                && t[4]->ne[0] == n_embd && t[4]->ne[1] == n_ff && t[1]->type == GGML_TYPE_F32 && t[3]->type == GGML_TYPE_F32;
        if (!ok) {
            LLAMA_LOG_ERROR("%s: %s: layer %d: missing tensor or shape mismatch (n_embd %lld, n_ff %lld)\n",
                            __func__, path, il, (long long) n_embd, (long long) n_ff);
            break;
        }
        ggml_tensor ** dst[5] = { &layers[il].w1, &layers[il].b1, &layers[il].w2, &layers[il].b2, &layers[il].down_t };
        for (int i = 0; i < 5; ++i) {
            *dst[i] = ggml_dup_tensor(ctx.get(), t[i]);
            ggml_set_name(*dst[i], t[i]->name);
            copies.emplace_back(*dst[i], t[i]);
        }
        ++n;
    }
    if (ok && n > 0) {
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(cpu)));
        ok = buf != nullptr;
        for (auto & c : copies) {
            if (ok) {
                ggml_backend_tensor_set(c.first, c.second->data, 0, ggml_nbytes(c.second));
            }
        }
    }
    gguf_free(g);
    ggml_free(src);
    if (!ok) {
        layers.clear();
        return -1;
    }
    keep = keep_;
    max_tokens = max_tokens_;
    relu = model.hparams.llm_ffn_op == LLM_FFN_RELU;
    this->model = &model;
    LLAMA_LOG_INFO("%s: %s: %d sparse FFN layers, keep %.3f, up to %d tokens per ubatch\n", __func__, path, n, keep, max_tokens);
    return n;
}

static bool is_plain_cpu(const ggml_tensor * t) {
    if (!t || !t->buffer) {
        return false;
    }
    // plain row-major host layouts: the CPU buffer and the mmap'd model file (not CPU_REPACK, AMX, KleidiAI, GPUs)
    const char * name = ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer));
    return strcmp(name, "CPU") == 0 || strcmp(name, "CPU_Mapped") == 0;
}

bool llama_ffn_sparse::active(const llama_layer & L, int il, int64_t n_tokens) const {
    if (il < 0 || il >= (int) layers.size() || !layers[il].w1 || n_tokens > max_tokens) {
        return false;
    }
    static bool warned = false;
    if (!is_plain_cpu(L.ffn_gate) || !is_plain_cpu(L.ffn_up)) {
        if (!warned) {
            LLAMA_LOG_WARN("%s: ffn_gate / ffn_up of layer %d are in buffer type %s, not a plain CPU buffer (offloaded "
                           "or repacked; use -ngl 0 --no-repack): dense FFN\n", __func__, il,
                           L.ffn_gate && L.ffn_gate->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(L.ffn_gate->buffer)) : "none");
            warned = true;
        }
        return false;
    }
    if (L.ffn_down_b) {
        return false;
    }
    for (const ggml_tensor * w : { L.ffn_gate, L.ffn_up, L.ffn_down }) {   // the sparse ops apply no weight scales
        if (model->w_scales.get(w) || model->w_scales.get_input(w)) {
            return false;
        }
    }
    return true;
}

ggml_tensor * llama_ffn_sparse::build(ggml_context * ctx0, const llama_layer & L, int il, ggml_tensor * cur) const {
    const layer & S = layers[il];
    const int64_t n_ff  = S.down_t->ne[1];
    const int64_t n_tok = cur->ne[1];
    const int k = std::max(1, std::min((int) n_ff, (int) (keep * n_ff + 0.5f)));
    cur = ggml_cont(ctx0, cur);
    ggml_tensor * h = ggml_relu(ctx0, ggml_add(ctx0, ggml_mul_mat(ctx0, S.w1, cur), S.b1));
    ggml_tensor * s = ggml_add(ctx0, ggml_mul_mat(ctx0, S.w2, h), S.b2);
    ggml_tensor * ids = ggml_top_k(ctx0, s, k);
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ggml_tensor * a1[4] = { cur, L.ffn_gate, L.ffn_up, ids };
    ggml_tensor * a = ggml_custom_4d(ctx0, GGML_TYPE_F32, k, n_tok, 1, 1, a1, 4, op_gateup, GGML_N_TASKS_MAX, relu ? (void *) 1 : nullptr);
    ggml_tensor * a2[3] = { a, S.down_t, ids };
    return ggml_custom_4d(ctx0, GGML_TYPE_F32, cur->ne[0], n_tok, 1, 1, a2, 3, op_down, GGML_N_TASKS_MAX, nullptr);
}
