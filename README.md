# llama.cpp-classifier

This is [llama.cpp](https://github.com/ggml-org/llama.cpp), kept in sync with upstream by merges, plus these patches:

- **Per-context early exit:** a classifier and text generation can share one loaded model.
- **LoRA training:** train a LoRA adapter in C/C++ on a frozen base model, including a quantized one, on CPU or GPU, and save it as a GGUF adapter that `--lora` loads.
- **Classifier heads** on the hidden states of the loaded model: yes/no (`noul`), multiple-choice (`choice`) and ordinal (`score`) questions. `llama-classifier` extracts features and trains heads. llama-server answers with them (`POST /classify`), answers questions given in the request without a head, returns pooled hidden states (`POST /features`), scores fixed answers by log-probability (`POST /options`) and trains new heads live (`POST /classify/train`).
- **llama-server LoRA training** on the loaded model between requests (`--lora-train`, `POST /lora/train`).
- **MoE expert routing for expert caching:** a selection bias toward chosen experts and a tail skip for non-resident ones (`--moe-selection-bias`, `--moe-resident`, `--moe-tail-skip`), for any MoE architecture.
- **Lazy MoE experts** (`--lazy-mode experts`): expert tensors are read from the file on demand, so a MoE model larger than RAM runs without thrashing the page cache.
- **PTQ1_0** (ggml type 143): PrismML's ternary type at group 128, ported from the [PrismML-Eng fork](https://github.com/PrismML-Eng/llama.cpp) with its authorship kept, so their Ternary Bonsai GGUFs load.
- **CPU + integrated GPU co-processing** of Q4_0 matrix products for prompt processing on boards whose GPU shares memory with the CPU (`GGML_CPU_COPROC_VULKAN`).
- **Bolmo / Bwen byte-level models** (arch `bolmo`): conversion, quantization and `llama-bolmo` for generation and per-byte scoring.
- **KV-cache fold** (`llama-kv-fold`): per-channel key scales and a key offset written into the model, exact for attention, so a 4.5-bit K cache loses about 7–12× less.
- **Smaller changes:** fixes to the server's `ngram-cache` drafter, Q8_1 input reuse and a per-fusion switch in CUDA, per-layer FFN widths for `qwen3`, embeddings output on models with an MTP block (`qwen4exp`, `deepseek4`), and a Vulkan `mul_mmq` fix. See [Other changes](#other-changes).

Everything else is unchanged upstream code. For building, models, tools and the full documentation, see the [official llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md).

## Early exit (classifier)

```c
#include "llama-ext.h"   // in src/

// n > 0: this context builds its graph only up to n trunk layers (then output_norm,
// pooling / LM head as usual). 0 = all layers (default).
void llama_set_n_layer_exit(struct llama_context * ctx, int32_t n);
```

The weights belong to `llama_model`, and a context owns only its KV cache and compute buffers. So one model can serve two contexts:

- **Generation context:** full depth, normal decoding.
- **Classifier context:** `embeddings = true`, `llama_set_n_layer_exit(ctx, L)`. Its graph stops after `L` layers and computes no LM head. That makes it much cheaper than a full forward pass.

To use it as a classifier:

1. Load the model once and create both contexts from it.
2. Read the residual after layer `L-1` (graph tensor `l_out-<L-1>`) with an eval callback (`cb_eval`), and mean-pool it over the item's tokens. You can also set `pooling_type = LLAMA_POOLING_TYPE_MEAN` and read the pooled embedding. That output passes through `output_norm`, so it is the normalised version of the same feature.
3. Train a linear head on those features (`llama-classifier`, below).
4. At run time, send items to the classifier context. The generation context's prompt prefill passes through the same layer, so a program using the C API can also classify a generation request from its own prefill at no extra cost. (llama-server's `/classify` always runs the input through the classifier context.)

The middle layers usually make the best classifier features. Pick `L` by cross-validation on your own data; `llama-classifier train` does that.

`common/classifier.h` holds the shared parts: the head file, the feature capture through `cb_eval`, and the fitting code.

## Classifier heads

A head reads the residual after `L` layers, pooled over the input (`mean`; `mean1`, the mean without the first token, whose attention-sink activation is large and content-independent; or the `last` token), and answers one question:

| type | model | answer |
|---|---|---|
| `noul` | logistic regression, `p = sigmoid(w·x + b)` | `{"type": "noul", "noul": 0.97}` |
| `choice` | softmax over K options, `softmax(W x + b)` | `{"type": "choice", "choice": "b", "confidence": 0.8, "probabilities": {"a": 0.1, "b": 0.8, "c": 0.1}}` |
| `score` | proportional odds over K ordered levels, `P(y ≤ k) = sigmoid(θ_k − w·x)` | `{"type": "score", "score": 2.4, "level": 2, "label": "high", "confidence": 0.6, "legend": {"0": "none", ...}, "probabilities": {"0": 0.02, ...}}` |

`score` is the expected level, `Σ k·p_k` (a float); `level` is the most likely level (an integer) and `label` its description, for callers that need a whole level. `confidence` is the probability of the most likely option or level. Every answer also carries the head's `version` when the head file has one.

**Head file** (GGUF, `general.type = classifier`):
- `classifier.layer` (u32): L
- `classifier.question_id` (str)
- `classifier.type`: `noul` | `choice` | `score` (default `noul`)
- `classifier.pooling`: `mean` | `mean1` | `last` (default `mean`)
- `classifier.options` ([str]): the option ids (`choice`) or level descriptions (`score`)
- `classifier.weight` F32 [n_embd, n_out] (n_out = K for `choice`, else 1)
- `classifier.bias` F32: [1] (`noul`), [K] (`choice`) or the K−1 non-decreasing thresholds θ (`score`)
- `classifier.version` (str, optional): an id for this fit, returned with every answer (default from `llama-classifier train`: `<question>-<UTC time>-<data hash>`; live training: `<question>-live-<UTC time>`)
- `classifier.model` (str, optional): the model the head was fitted on (description, parameter count, file size); llama-server and `llama-classifier eval` warn when a head is used with a different model or quantization
- `classifier.info.*` (str): training settings and cross-validated metrics, written by the trainers (`train` also writes `data_fnv1a`, the hash of the training file, and `prompt_hash` when given)

**Training** minimises the mean log-loss plus `‖w‖² / (2·C·n)`, on standardised features, with L-BFGS. That is the same objective as scikit-learn's `LogisticRegression(C)`; the stored weights are for raw features. On a 579-item, 4096-dimensional set it matches scikit-learn's probabilities within 2e-4, for `noul` and for `choice` (multinomial). `--balanced` weights the classes by inverse frequency.

```sh
# data: one JSON object per line, {"text": "...", "label": ...} or {"tokens": [...], "label": ...}
# labels: true/false (noul), an option id (choice), a level name or index 0..K-1 (score)
llama-classifier train -m model.gguf --data train.jsonl --type choice --question-id topic --out topic.gguf \
    [--options a,b,c] [--layers auto|L,L,..] [--pooling auto|mean|last|mean1] [--C 0.01,0.1,1] [--folds 5] [--balanced] \
    [--version ID] [--prompt-hash STR] [--grid-out grid.tsv]
llama-classifier eval     -m model.gguf --head topic.gguf --data test.jsonl [--predictions out.jsonl]
llama-classifier features -m model.gguf --data data.jsonl --layers 12,18 --out feats    # raw f32 matrices + labels
llama-classifier fit      --features feats.L18.mean.f32 --labels feats.labels.txt --layer 18 --type noul --out head.gguf
```

`train` extracts the features of every candidate layer and all three poolings in one pass. It cross-validates every layer × pooling × C setting (stratified k-fold), picks the one with the lowest held-out log-loss, and refits it on all items. `--layers auto` tries about 12 layers spread over the depth.

**One layer for several heads.** Heads that read the same layer share one early-exit context in llama-server; heads on different layers each need their own (KV and compute buffers). To keep several heads on one layer, train each with `--grid-out`, then `llama-classifier pick-layer --grids a.tsv,b.tsv,..` prints, per layer, each head's best held-out log-loss and their sum, and picks the layer with the lowest sum; retrain each head with `--layers <that layer>`.

## LoRA training

```sh
llama-finetune -m base.gguf -f train.txt -c 256 -b 256 -ub 256 \
    --lora-rank 8 --lora-alpha 16 [--lora-targets attn_q,attn_v] -lr 1e-4 -epochs 1 -o adapter.gguf
llama-cli -m base.gguf --lora adapter.gguf      # or llama-server --lora, llama-export-lora to merge
```

- **What gets adapted:** `--lora-rank` trains a new adapter on the attention and FFN matrices of every layer, with the base model frozen. The base can be F32 or quantized (tested: F32 and Q4_K_M), on CPU or GPU. On the GPU, quantized weights are dequantized for the backward pass.
- **Initialisation:** as PEFT does it. A is uniform in ±1/sqrt(n_in) and B = 0, so the adapter starts as a no-op. The output is scaled by alpha / rank.
- **API** (`src/llama-ext.h`):
  - `llama_adapter_lora_init_trainable()` creates the adapter.
  - `llama_set_adapters_lora()` attaches it to a context.
  - `llama_opt_init()` with `llama_opt_param_filter_lora` sets up training of only the adapter.
  - `llama_opt_epoch()` trains, and `llama_adapter_lora_save()` writes the adapter.
- **Checked against PyTorch** (Qwen2.5-0.5B in F32, same initial adapter, same 256-token windows, same optimizer):
  - With SGD, the adapter weights after two steps agree to 0.2%.
  - With AdamW, the losses agree to about 1e-4 for the first 20 steps. After that, ordinary float32 differences grow, as they also do between two PyTorch runs started 1e-6 apart.

- **Example result:** Qwen2.5-0.5B Q4_K_M base, 8k tokens of synthetic disassembly, rank 8, 1 epoch, lr 2e-4. Perplexity on held-out text from the same source fell from 6.60 to 5.43. With 2 epochs at lr 1e-3 it overfits (held-out 9.26).

It also fixes two problems in upstream training code:

- **ggml-opt (correctness):** with dynamic graphs, which `llama_opt_epoch` uses, the gradients were never reset between optimizer steps, so every step applied the sum of all gradients so far. This affected full fine-tuning too.
- **CPU `OUT_PROD` with quantized weights (speed):** each weight row was dequantized once per token. Doing it once per row and thread gives a 5.6× faster training step on a Q4_K_M base, with bit-identical results.

**Limits:**
- There's no backward pass for flash attention (it's switched off during training), for `MUL_MAT_ID` (no MoE), or for the Gated DeltaNet / SSM ops.
- The GELU family has a backward pass in this fork: GELU (tanh form), GELU-erf and GELU-quick, plus their split GEGLU forms, built from existing ops (erf via the Abramowitz–Stegun approximation, error 1.5e-7), so Gemma-family models train. Gemma 3 270M F32 matches PyTorch within 2e-4 in loss and 0.3% in the trained weights after two SGD steps, on CPU and GPU.
- Training needs `n_ubatch == n_ctx`, or the K and V projections get no gradient.

## llama-server

```sh
llama-server -m model.gguf --classifier head.gguf --lora-train
```

### Classifier

```sh
llama-server -m model.gguf --classifier topic.gguf [--classifier h2.gguf] [--features] [--classifier-train [--classifier-dir DIR]]
```

Heads that read the same layer share one early-exit context on the loaded model.

**Memory:** each distinct layer costs one context of `--classifier-ctx` tokens (default 4096), and so does `--features`. Its KV cache covers all layers, because early exit doesn't shrink it. Its compute buffer is sized by `--classifier-ubatch` (default 512 tokens), not by the longest input: the pooled features are accumulated over compute steps, so an input or a batch longer than that is processed in several steps with the same answers. On Qwen3-8B at `--classifier-ctx 4096` the buffer is 341 MiB at the default against 2,726 MiB with one step per input. Lower `--classifier-ctx` when inputs are short; raise `--classifier-ubatch` only if long inputs need fewer steps and the memory is there.

**`POST /classify`** with `{"input": "text"}`, a token array, or an array of either. One answer per head:

```json
{"model": "...", "answers": {"relevant": {"type": "noul", "noul": 0.97}, "topic": {"type": "choice", ...}}, "usage": {"input_tokens": 35, "output_tokens": 0}}
```

An array of inputs returns an array of these objects, in input order. The inputs are decoded together, one sequence each, in batches of up to 64 inputs whose tokens fit `--classifier-ctx` together; on Qwen3-8B (CPU) 24 short inputs take 1.7× less time than one request each. Hybrid and recurrent models (e.g. Qwen3.5) still decode the inputs one by one. Requests from different callers are batched too: when the server takes a `/classify` request for the heads, it also takes the other head-only `/classify` requests already waiting and decodes their inputs together (up to 64 inputs that fit `--classifier-ctx`), so callers that send one input per request still get batches (16 concurrent single-input requests on Qwen3-8B, CPU: 1.7× faster than one after another). Answers match the one-by-one ones up to the float noise of a different batch shape.

**Span pooling** (a term in the request): a head with pooling `span` reads the mean over the tokens of a term in the input, widened by `classifier.span_window` tokens on each side (default 2), instead of the whole input: for questions about one mention (how central a name is to an article) that a whole-input mean washes out. `/classify` takes `"term"`, one string for every input or one per input; the term is found case-insensitively in the decoded input, and a span head answers with `span_tokens` (0 = the term was not found and the head used the `mean1` pooling instead). Train such heads with `llama-classifier train` on JSONL items that carry `"term"` (`--pooling span`, `--span-window N`, or `auto`, which then also tries `span`). Heads with other poolings answer in the same request, so one pass gives a whole-article head (e.g. sentiment) and a mention head (e.g. relevance). Live training takes them too: `"term"` per `data` item, `"pooling": "span"` (or `auto`, which adds `span` when items have terms) and `"span_window"`.

**Questions in the request** (`--features`): `/classify` answers questions given in the request, with no trained head. The input and the answers are pooled at one layer and compared by cosine similarity. It runs in the classifier's own contexts and does not touch the generation slots.

A question is either:
- `{"type": "noul", "true": <answer>, "false": <answer>}`
- `{"type": "choice", "options": {"<name>": <answer>, ...}}`

An answer is a description (a string) or example texts (an array of strings; their pooled vectors are averaged). Examples work much better than descriptions on fine decisions.

```json
{"input": ["text 1", "text 2"],
 "questions": {"keep": {"type": "noul",
                        "true":  ["example kept 1", "example kept 2"],
                        "false": ["example dropped 1", "example dropped 2"]}},
 "questions_only": true}
```

Each answer has `noul` (or `choice` and `probabilities`), the raw `similarity` per answer and `"method": "similarity"`.

Request fields:

| field | values | default | what it does |
|---|---|---|---|
| `layer` | 0..n_layer | 2/3 of the depth | the layer read; 0 is the fast mode, see below |
| `pooling` | `mean1`, `mean`, `last`; at layer 0 `idf`, `mean` | `mean1`; at layer 0 `idf` | how the tokens are pooled (`mean1` skips the first token) |
| `template` | `echo`, `prompteol`, `none` | `echo` | how the text is presented, see below |
| `unlabeled` | array of texts, or `"inputs"` | none | removes the main shared direction, see below |
| `attention` | `causal`, `bidirectional` | `causal` | attention mask, see below |
| `temperature` | > 0 | 0.05 | softmax over the similarities |
| `questions_only` | bool | false | skip loaded heads |

Templates:
- `echo`: the model reads `Rewrite the following paragraph: <text>\nThe rewritten paragraph: <text>` and the mean is taken over the second copy only. Every token there has already seen the whole text. Costs twice the tokens.
- `prompteol`: the model reads `This sentence: "<text>" means in one word: "`. With `"pooling": "last"` the final token is read, with `mean` the text.
- `none`: the text as is.

`unlabeled`: give 50–100 texts of the same kind as the inputs, or `"inputs"` to use all inputs of the request. Their top principal direction is removed from inputs and answers before the cosine. Texts of another kind do not work.

Fast mode, `"layer": 0`: no forward pass. Each text (input, description or example) is the average of its tokens' input embedding rows, weighted by log(N / document frequency) over all texts of the request (`idf`), or unweighted (`mean`). It needs no classifier context: start the server with `--classifier-fast` (or any other classifier option). All inputs of a request are answered together, so the weights count every input; `template` and `attention` do not apply. With a handful of examples per answer it can match the hidden-state questions on a fine decision; with descriptions only, or where word order decides, read a layer instead.

`attention: "bidirectional"`: every token sees the whole input, as in an encoder. The model was trained causally, so this usually costs some accuracy; it is there to try. Each input must fit in `--classifier-ubatch` tokens. Trained heads always run causally.

`/features` takes `template` and `attention` too (default `none` and `causal`). Span pooling cannot be combined with a template.

**`POST /options`** (`--options`): the model's own choice among a few fixed answers, with no head and no training. The body: `prompt` (a string), `options` (2–32 strings), `separator` (default a space). Each option is scored by the summed log-probability of its tokens after the prompt (the same scoring as an option log-prob multiple-choice test); the prompt is decoded once and shared by the options, each option in its own sequence. The response: per option `logprob`, `n_tokens` and `probability` (softmax over the options), plus `best`, `confidence` and `prompt_tokens`. It runs the full model depth in its own context of `--classifier-ctx` tokens, separate from the generation slots.

**`POST /features`** (`--features`): pooled hidden states, for training heads elsewhere. The body: `input` as above, `layers` (a number or an array, 1..n_layer, default the last layer), `pooling` (`"mean"`, `"last"` or both, default both). The response: `{"model": ..., "n_tokens": 35, "features": {"18": {"mean": [...], "last": [...]}}}`. The context computes only up to the deepest requested layer. An array of inputs is decoded in batches like `/classify` (one sequence per input) and returns an array in input order.

**`POST /classify/train`** (`--classifier-train`, which also turns on `--features`): fits a head on labelled inputs and adds it to `/classify` at once. A head with the same `question_id` is replaced. The body:
- `question_id`: letters, digits, `_`, `-`, `.`
- `type`: `noul` (default), `choice` or `score`; `options` as in `llama-classifier` (required for `score`)
- `data`: `[{"input": text or tokens, "label": ...}, ...]`; inputs longer than `--classifier-ctx` are cut
- `layers` (`"auto"` or numbers), `pooling` (`auto`, `mean`, `last`), `C` (a number or an array, default 0.01, 0.1, 1), `folds` (5), `seed`, `balanced`
- `save`: write `<question_id>.gguf` to `--classifier-dir`

The response has the chosen `layer`, `pooling` and `C`, the cross-validated metrics (`cv`: log-loss, accuracy, AUC for `noul`, MAE for `score`), one line per tried setting (`grid`), the saved path and the list of heads. The features are computed in the server loop one input at a time, so other requests are served in between; the cross-validation and the fit run on the HTTP thread. The result is identical to `llama-classifier train` on the same data and settings.

These endpoints also work on a native decision model; its own decision endpoint gives the same answers with them turned on.

### LoRA training

`POST /lora/train` starts a job on the loaded model. The body fields:
- `text` or `tokens`
- `rank`, `alpha` (default 2·rank, unlike `llama-finetune`, where the default is rank), `lr`, `epochs`, `n_ctx` (a multiple of 256), `targets`, `name`, `seed`
- `file`: a plain file name inside `--lora-train-dir` (default `<temp dir>/llama-lora-train`)
- `register`: `false` saves the adapter without adding it to the server
- `priority`:
  - `idle` (default): train only while no request is being processed
  - `shared`: also train between decode rounds

Training runs in its own context, one optimizer step per server-loop iteration.

**When the job finishes:**
- The adapter is saved to the training dir. A registered adapter is reloaded from that file after the server sleeps, so keep the file.
- It is added to `/lora-adapters` with scale 0. A request uses it with `"lora": [{"id": <adapter_id>, "scale": 1}]`.

`GET /lora/train` lists the jobs (step, losses, adapter id). `POST /lora/train/cancel` with `{"id": n}` cancels one.

With `--lora-train`, weight repacking is turned off.

Recurrent, hybrid (SSM / DeltaNet), MoE and diffusion models are rejected: some of their ops have no backward pass.

The training context (F32 KV cache, AdamW state, the backward graph) needs memory on top of the served model.

## CPU + GPU co-processing (`GGML_CPU_COPROC_VULKAN`)

On a board whose integrated GPU shares memory with the CPU, the GPU can compute part of each prompt-sized Q4_0 matrix product while the CPU threads compute the rest. Decode is bound by memory bandwidth, which the two share, so only prompt processing gains.

- Build the CPU backend with `-DGGML_CPU_COPROC_VULKAN=ON` (needs the Vulkan loader and headers, `glslc`, and the CPU repack path, which is on by default). It works without the Vulkan backend.
- `GGML_CPU_COPROC=<fraction>` turns it on: the starting share of each Q4_0 weight's rows (in blocks of 128) that the GPU computes. When the repack buffer loads the weights, up to `GGML_CPU_COPROC_MAX` of the rows (default: the share + 0.15) are also copied into GPU buffers.
- The share adapts per weight: one block more when the GPU finished before the CPU, one block fewer when the CPU waited more than 3% of the product's time. This follows the CPU's speed as it throttles. `GGML_CPU_COPROC_ADAPT=0` keeps the share fixed.
- `GGML_CPU_COPROC_MIN_N` (default 32): only products with at least this many tokens use the GPU. `GGML_CPU_COPROC_DEVICE` picks the Vulkan device. `GGML_CPU_COPROC_STATS=1` prints timing totals at exit.
- The GPU kernel (`ggml/src/ggml-cpu/coproc-q4_0.comp`) uses the packed int8 dot product (Vulkan 1.3 `shaderIntegerDotProduct`) on Q4_0 weights and Q8_1 activations.

Measured with Qwen3-1.7B Q4_0, 4 threads, prompt 512 tokens, on a 4-core single-board computer whose integrated GPU uses the Mesa v3dv driver, no fan: 92.0 tok/s on the CPU alone, 104.5 tok/s with `GGML_CPU_COPROC=0.3` (+13.5%); perplexity unchanged within its error. That driver needs a fix to run compute workgroups concurrently: Mesa leaves the dispatch's maximum supergroup ID at 0, so its workgroups run one at a time and the GPU is about 4x slower.


## Bolmo / Bwen byte-level models (arch `bolmo`)

[Bolmo](https://huggingface.co/allenai/Bolmo-1B) and Bwen (Minixhofer et al., Nature 2026) turn a subword LLM into a byte-level one. The subword transformer becomes the "global" model over patches. Around it: a one-layer mLSTM encoder over bytes (byte embedding plus the embedding of the longest subword ending at each byte), a boundary predictor with one byte of lookahead, last-byte pooling into patches, and a four-layer mLSTM decoder with a 520-way head (byte, or byte with a fused patch boundary).

- `convert_hf_to_gguf.py` converts `BolmoForCausalLM` checkpoints (OLMo 2 global blocks for Bolmo-1B/7B, Qwen 3 / Llama 3 pre-norm blocks for Bwen-8B). It needs the source subword tokenizer (`original_identifier` in the config) to store the suffix table; `BOLMO_SUBWORD_TOKENIZER=<path>` points to a local copy. `llama-quantize` works as usual.
- `llama_decode` on a bolmo context runs only the global transformer: the batch carries patch embeddings, and the context returns the decoder's normalised input per patch as embeddings (`embeddings = true`, pooling none).
- `include/llama-bolmo.h` runs the whole model: `llama_bolmo_prefill` / `llama_bolmo_step` follow `BolmoForCausalLM.generate()` (the prompt's last byte is force-decoded to decide whether it ends a patch; the global model steps only when a byte ends a patch), and `llama_bolmo_score` is the teacher-forced forward.
- `llama-bolmo -m model.gguf -p "..." [-n N] [--temp T] [--show-patches]` generates; `--score-file text.txt` prints the per-byte NLL (marginalised over the boundary flag) as bits/byte; `--dump-logits` writes the teacher-forced logits and boundaries; `--mcq items.tsv out.tsv` scores multiple-choice options byte by byte, as generation would see them (`--mcq-tf`: teacher-forced, with the predictor's one byte of lookahead; `--mcq-exact`: summed over both boundary choices at every byte, which the other two bracket). `llama_bolmo_snapshot_take/restore` save and restore the sequence state for such searches.
- `llama-bolmo --server [--host H] [--port P]`: `POST /v1/completions` (prompt, max_tokens in bytes, temperature, seed, stop strings), `POST /score` (bits/byte of a text) and `GET /health`, one request at a time.
- The mLSTM cell is built from standard ggml ops, so it runs on any backend: the parallel form for a prompt, the recurrent step for generation. It drops the kernels' max-stabiliser, which is safe in float32 because the gates are soft-capped (input gate at most e^15, forget gate at most 1). One sequence at a time.

Checked against the Hugging Face implementation (transformers 4.57.3, xlstm 2.0.4, CPU fp32) on four prompts (English, x86 assembly, Python): Bolmo-1B F32 and Bwen-8B F16 GGUFs give the same patch boundaries, a 520-way softmax within KL 4e-6 at every byte (2.4e-5 with CUDA), and byte-identical greedy generations (51 to 91 bytes). Bolmo-1B Q8_0 on an RTX 5090: 660 bytes/s greedy generation.

## KV-cache fold (`llama-kv-fold`)

A quantized K cache loses most of its accuracy to a few key channels that are much larger than the rest, and to the
keys' common offset. The fold removes both without changing attention: for each attention layer, key head and channel,

- the key is stored as `(k - shift) / fold`, after RoPE
- the queries of that key head are multiplied by `fold`

`q.k` then changes only by `q.shift`, which is the same for every key of a query and cancels in the softmax, and `fold`
is equal within each RoPE pair, so it commutes with RoPE. With an F16 cache the folded model gives the same logits as
the original (KL 0.00000 on the test below).

`llama-kv-fold` calibrates the two tensors on a text and writes a copy of the model that carries them:

```sh
llama-kv-fold -m model.gguf -f calib.txt -o model-kvfold.gguf [-c 512] [--chunks 64] [--alpha 0.5] [--no-shift] [-t N]
```

`shift` is the mean key after RoPE, `fold` = rms_k^alpha / rms_q^(1-alpha) per channel, averaged within each RoPE pair;
the first token of each chunk (the attention sink) is left out. The tensors (`blk.N.attn_k_fold`, `blk.N.attn_k_shift`,
F32 `[n_embd_head_k, n_head_kv]`) are optional and loaded for any architecture; a model without them runs as before.
Use the folded model with a quantized cache as usual (`-ctk q4_0 -ctv q4_0`, `-ctk iq4_nl -ctv iq4_nl`). llama.cpp's
own Hadamard rotation of quantized caches stays on; the fold is applied before it.

Qwen3-1.7B F16, 512-token chunks of a held-out text, KL to the F16 model with an F16 cache, CPU, calibration on 32k
tokens of a disjoint text:

| K / V cache | bits per value | model | mean KL | same top token |
|---|---:|---|---:|---:|
| q4_0 / q4_0 | 4.5 | original | 0.244 | 84.2% |
| q4_0 / q4_0 | 4.5 | fold only (`--no-shift`) | 0.050 | 93.1% |
| q4_0 / q4_0 | 4.5 | fold + shift | 0.033 | 94.3% |
| iq4_nl / iq4_nl | 4.5 | original | 0.315 | 81.9% |
| iq4_nl / iq4_nl | 4.5 | fold + shift | 0.025 | 94.9% |
| q4_0 / q8_0 | 4.5 / 8.5 | original | 0.237 | 84.6% |
| q4_0 / q8_0 | 4.5 / 8.5 | fold + shift | 0.028 | 94.8% |
| q5_0 / q4_0 | 5.5 / 4.5 | original | 0.045 | 93.6% |
| q5_0 / q4_0 | 5.5 / 4.5 | fold + shift | 0.012 | 96.5% |
| q5_0 / q5_0 | 5.5 | original | 0.043 | 93.8% |


## MoE expert routing for expert caching

When only some experts fit in fast memory (VRAM, or RAM for a model larger than RAM), these options steer the router
toward them. Both work on any MoE architecture and are ignored, with a warning, on dense models.

- `--moe-selection-bias FILE`: a raw float32 file of `n_layer × n_expert` values, added to the router probabilities
  for choosing the experts only; the weights of the chosen experts are unchanged. It adds to a model's own selection
  bias when it has one. A file with a positive value on a set of experts and 0 elsewhere favours that set; a file
  that keeps only a set (large negative values elsewhere) comes close to a copy of the model pruned to it (per-token
  KL 0.009 between the two on a 512-expert model).
- `--moe-resident FILE --moe-tail-skip N`: of the experts a token selects, those at rank ≥ N (0-based) that are not
  resident (value ≤ 0 in the file) get weight 0. The other weights are not renormalised.

## Lazy MoE experts (`--lazy-mode experts`)

`--lazy-mode experts` maps every MoE expert tensor and reads it from the file on demand, with the kernel's normal
readahead. With `--load-mode none` the other weights are read into RAM at load, so the page cache holds only experts.
Lazily read experts are not repacked, so the CPU runs them as with `--no-repack` (KL 0.000001 against a `--no-repack`
reference).

A 501B MoE model of 134 GB on a machine with 125 GB of RAM, experts on the CPU: 3.6 s per token, against 31.6 s and a
39-minute load with the default mmap, which thrashes.

## PTQ1_0 ternary type

`PTQ1_0` (ggml type 143) is PrismML's ternary format at group 128, ported from the
[PrismML-Eng fork](https://github.com/PrismML-Eng/llama.cpp) with the original authors kept on the commits, so its
Ternary Bonsai GGUFs load here. Ported: the codec and the generic CPU dot product, x86 SSE2/SSSE3, ARM NEON + i8mm,
CUDA dequantization and matrix-vector products. Not ported: `PQ2_0` (type 142), Metal and Vulkan.

Ternary-Bonsai-2-2B perplexity: 31.82 on CPU and 31.97 on CUDA, against 31.79 / 31.84 for its `Q2_0` file. On CUDA,
prompt batches run about 5× slower than `Q2_0`, which has the quantized matrix-matrix kernels that `PTQ1_0` lacks here.

## Other changes

- **`ngram-cache` speculative drafter in llama-server:** the context cache is reset per request (a slot used to keep
  the previous request's n-grams, so a shorter new prompt was never indexed and drafted nothing), n-grams across the
  update boundary are indexed, and the draft length follows `--spec-draft-n-max` instead of a fixed 8.
- **CUDA:** several matrix-vector products often read the same single-column input (Q, K, V and gates); the first one's
  Q8_1 quantization is now kept and reused until a node writes over the input (single stream only;
  `GGML_CUDA_NO_Q8_REUSE=1` turns it off). `GGML_CUDA_FUSION_SKIP=name,...` turns single kinds of fusion off for A/B
  tests (`shexp`, `moered`, `gdn`, `ssmscan`, `topk`, `snake`, `multi`, `glu`, `mmscale`, `mmadd`, `rmsnorm`, `ssmconv`,
  `unary`, `dsv4hc`).
- **`qwen3` per-layer FFN width:** `qwen3.feed_forward_length` may be an array, so a file whose FFN layers have
  different widths (after width pruning) loads.
- **Embeddings on `qwen4exp` and `deepseek4`:** the MTP hidden-state width (`llama_model_n_embd_nextn`) is separate from
  the embeddings output width, so embeddings readouts (`--embeddings`, `llama-embedding`) no longer read past the end of
  the output and abort.
- **Vulkan `mul_mmq`:** the compute loop no longer runs k steps past the end of the matrix, which multiplied stale shared
  memory and returned NaN on drivers that keep earlier shared-memory contents (Raspberry Pi 5, v3dv).
