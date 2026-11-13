#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Early exit: n > 0 builds the trunk graph only up to n layers (then output_norm, pooling /
// LM head as usual), e.g. for a classifier context sharing the model's weights. 0 (or n >= n_layer)
// = all layers. The memory (KV / recurrent state) of layers >= n is not updated while n is set, so when
// n rises (more layers will run) on a context whose memory holds tokens, the memory is cleared (with a
// warning); lowering n keeps it.
LLAMA_API void llama_set_n_layer_exit(struct llama_context * ctx, int32_t n);

// Per-layer expert selection bias for MoE layers whose model has no bias of its own: bias[il * n_expert + e] is
// added to the router probabilities of layer il for choosing the top-k experts only (the expert weights stay the
// unbiased probabilities), as DeepSeek-V3's exp_probs_b. bias = NULL clears it. Returns 0, or -1 if the sizes do not
// match the model.
LLAMA_API int32_t llama_set_moe_selection_bias(struct llama_context * ctx, const float * bias, int32_t n_layer, int32_t n_expert);

// Tail skip for MoE layers: of the top-k experts a token selects, those at rank >= rank_from (0-based, by router
// probability) that are not resident (resident[il * n_expert + e] <= 0) get weight 0 (the others keep their weights,
// not renormalised), as Strata's route tail skip. resident = NULL or rank_from < 0 turns it off. Returns 0 or -1.
LLAMA_API int32_t llama_set_moe_tail_skip(struct llama_context * ctx, const float * resident, int32_t n_layer, int32_t n_expert, int32_t rank_from);

// Residual cut after layer il (where control vectors are added): a = U^T (h - mu), h -= U (a * (|a| < thr)), i.e. the
// component along k orthonormal directions is projected out (centred on mu) only for tokens whose coordinate lies
// within thr; outlier tokens keep it (thr = FLT_MAX gives a plain centred projection). U holds the k directions one
// after another (U[j * n_embd + i]), mu n_embd values, thr k values. One call per layer; U = NULL clears every layer.
// Returns 0, or -1 if the sizes do not match the model.
LLAMA_API int32_t llama_set_resid_cut(struct llama_context * ctx, int32_t il, const float * U, const float * mu, const float * thr, int32_t n_embd, int32_t k);

// A separate threshold (k values) for tokens at sequence position 0 (the attention sink) under layer il's residual
// cut, e.g. 0 to spare the sink along a direction and FLT_MAX to project it; thr0 = NULL removes it. Call after
// llama_set_resid_cut for that layer. Returns 0 or -1.
LLAMA_API int32_t llama_set_resid_cut_first(struct llama_context * ctx, int32_t il, const float * thr0, int32_t k);

// Directional clamp after layer il: a = U^T (h - mu), then each of the k coordinates is bounded to [cmin, cmax] and the
// bounded part written back scaled by cstr, h += U ((clamp(a, cmin, cmax) - a) * cstr). Unlike steering, which shifts
// every token equally, this leaves a token inside the band untouched and only pulls back tokens whose projection runs
// past it. U, mu as in llama_set_resid_cut; cmin, cmax, cstr hold k values each. U = NULL clears every layer. 0 / -1.
LLAMA_API int32_t llama_set_resid_clamp(struct llama_context * ctx, int32_t il, const float * U, const float * mu, const float * cmin, const float * cmax, const float * cstr, int32_t n_embd, int32_t k);

// Dynamic, activation-conditioned steer after layer il: the residual's own projection on a probe decides the strength,
// h += sigmoid(sharp (probe . h - sthr)) * steer, per token. probe and steer hold n_embd values; sharp (gate sharpness)
// and sthr (gate threshold) are scalars. probe = NULL clears every layer. Returns 0 or -1 if the sizes do not match.
LLAMA_API int32_t llama_set_resid_dyn(struct llama_context * ctx, int32_t il, const float * probe, const float * steer, float sharp, float sthr, int32_t n_embd);

// LoRA training: a new adapter with trainable F32 tensors. For every 2D weight of the repeating
// layers whose name matches one of the targets ("blk.N.<target>.weight"), A [n_in, rank] is
// initialised like PEFT (uniform in +-1/sqrt(n_in)) and B [rank, n_out] is zero, so the adapter
// starts as a no-op. Its output is scaled by alpha / rank, as for loaded adapters.
// Set it on the context (llama_set_adapters_lora), then train it with llama_opt_init using
// llama_opt_param_filter_lora, and save it with llama_adapter_lora_save.
struct llama_adapter_lora_train_params {
    int32_t      rank;    // > 0
    float        alpha;   // 0 = rank (scale 1)
    const char * targets; // comma-separated, NULL = "attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down"
    uint32_t     seed;
};

LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init_trainable(
        struct llama_model * model,
        struct llama_adapter_lora_train_params params);

// write the adapter as a GGUF LoRA (loadable with llama_adapter_lora_init / --lora)
LLAMA_API bool llama_adapter_lora_save(const struct llama_adapter_lora * adapter, const char * path);

// param filter for llama_opt_init: only the tensors of the adapters set on the context.
// Note: llama_opt_init offers the F32 tensors of the adapters set on the context to any filter, so with
// llama_opt_param_filter_all a loaded F32 adapter is trained along with the model.
// The parameter flags are cleared again when the training context is freed.
LLAMA_API bool llama_opt_param_filter_lora(const struct ggml_tensor * tensor, void * userdata);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

//
// model/context data extraction
//

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLAMA_API bool llama_model_classifier_free_guidance(const struct llama_model * model);
