#pragma once

// Per-token sparse FFN (wllm fork, 2026-11-27): a learned predictor picks k of each layer's n_ff units per token, and
// only those rows of ffn_gate / ffn_up / the transposed ffn_down are read (ggml-cpu/ffn-sparse.cpp). Loaded from a
// side GGUF (llama_set_ffn_sparse): per layer il fsp.w1.<il> [n_embd -> r], fsp.b1.<il> [r], fsp.w2.<il> [r -> n_ff],
// fsp.b2.<il> [n_ff], fsp.down_t.<il> [n_embd, n_ff] (row u = column u of ffn_down). CPU only; used for ubatches of
// at most max_tokens tokens (decode), dense otherwise, and dense where the model's FFN weights are not in a plain CPU
// buffer (GPU offload, or repacked CPU_REPACK weights whose rows are interleaved).

#include "ggml-cpp.h"

#include <cstdint>
#include <vector>

struct llama_model;
struct llama_layer;

struct llama_ffn_sparse {
    struct layer {
        ggml_tensor * w1 = nullptr, * b1 = nullptr, * w2 = nullptr, * b2 = nullptr, * down_t = nullptr;
    };
    std::vector<layer> layers;   // per model layer; w1 == nullptr: dense
    float   keep       = 0.5f;
    int32_t max_tokens = 4;
    bool    relu       = false;   // gated ReLU FFN (qwen3.hidden_activation = relu) instead of SwiGLU
    ggml_custom_op_t op_gateup = nullptr;
    ggml_custom_op_t op_down   = nullptr;
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;

    // returns the number of sparse layers, or -1
    int32_t load(const llama_model & model, const char * path, float keep, int32_t first_layer, int32_t max_tokens);

    bool active(const llama_layer & L, int il, int64_t n_tokens) const;

    // cur: the FFN input after ffn_norm, F32 [n_embd, n_tokens]; returns the FFN output [n_embd, n_tokens]
    ggml_tensor * build(ggml_context * ctx0, const llama_layer & L, int il, ggml_tensor * cur) const;
};
