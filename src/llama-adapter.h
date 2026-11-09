#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    // first: F32 [1, n_tokens], 1 for tokens at sequence position 0 (only used when needs_first(il))
    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il, ggml_tensor * first = nullptr) const;

    bool needs_first(int il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

    // residual cut after layer il (see llama_set_resid_cut); U = nullptr clears every layer
    bool set_cut(
            const llama_model & model,
            int32_t il,
            const float * U,
            const float * mu,
            const float * thr,
            int32_t n_embd,
            int32_t k);

    // a separate threshold for tokens at sequence position 0 (see llama_set_resid_cut_first)
    bool set_cut_first(int32_t il, const float * thr0, int32_t k);

    // directional clamp after layer il (see llama_set_resid_clamp): along each of k directions,
    // bound the (centred) projection a = U^T (h - mu) to [cmin, cmax] and write the bounded part
    // back scaled by cstr: h += Ut ((clamp(a, cmin, cmax) - a) * cstr). U = nullptr clears clamps.
    bool set_clamp(
            const llama_model & model,
            int32_t il,
            const float * U,
            const float * mu,
            const float * cmin,
            const float * cmax,
            const float * cstr,
            int32_t n_embd,
            int32_t k);

    // dynamic, activation-conditioned steer after layer il (see llama_set_resid_dyn): the residual's
    // own projection on a probe gates an additive steer, h += sigmoid(sharp (probe . h - sthr)) * steer.
    // probe = nullptr clears the dynamic steer of every layer.
    bool set_dyn(
            const llama_model & model,
            int32_t il,
            const float * probe,
            const float * steer,
            float sharp,
            float sthr,
            int32_t n_embd);

private:
    bool init(const llama_model & model);

    struct cut_layer {
        ggml_tensor * U   = nullptr; // [n_embd, k]: row j = direction j
        ggml_tensor * Ut  = nullptr; // [k, n_embd]
        ggml_tensor * mu  = nullptr; // [n_embd]
        ggml_tensor * thr = nullptr; // [k]
        ggml_tensor * dthr0 = nullptr; // [k]: threshold at position 0 minus thr (optional)
        std::vector<float> thr_host;
        ggml_context_ptr       ctx0;
        ggml_backend_buffer_ptr buf0;
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_context_ptr       ctx;
        ggml_backend_buffer_ptr buf;
        // directional clamp (independent of the threshold cut above; own projection geometry)
        ggml_tensor * cU   = nullptr; // [n_embd, k]
        ggml_tensor * cUt  = nullptr; // [k, n_embd]
        ggml_tensor * cmu  = nullptr; // [n_embd]
        ggml_tensor * cmin = nullptr; // [k]
        ggml_tensor * cmax = nullptr; // [k]
        ggml_tensor * cstr = nullptr; // [k]
        ggml_context_ptr       ctxc;
        ggml_backend_buffer_ptr bufc;
        // dynamic activation-conditioned steer
        ggml_tensor * probe = nullptr; // [n_embd, 1]
        ggml_tensor * steer = nullptr; // [n_embd]
        ggml_tensor * sthr  = nullptr; // [1]
        float sharp = 1.0f;
        ggml_context_ptr       ctxd;
        ggml_backend_buffer_ptr bufd;
    };
    std::vector<cut_layer> cuts; // per layer, empty when no cut is set

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;
