#include "server-classifier.h"

#include "../../src/llama-ext.h" // staging API: llama_set_n_layer_exit

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// one early-exit context and the capture of its tapped layers
struct server_classifier_ctx {
    llama_context *           ctx = nullptr;
    common_classifier_capture cap;

    ~server_classifier_ctx() {
        llama_free(ctx);
    }
};

server_classifier::server_classifier()  = default;
server_classifier::~server_classifier() {
    if (opt_ctx) {
        llama_free(opt_ctx);
    }
}

bool server_classifier::features_enabled() const {
    return feat != nullptr;
}

static std::unique_ptr<server_classifier_ctx> make_ctx(llama_model * model, int32_t n_ctx, int32_t n_ubatch, int32_t n_threads,
                                                       int32_t n_embd, const std::vector<int32_t> & layers, int32_t exit,
                                                       int32_t n_seq = 1) {
    auto c = std::make_unique<server_classifier_ctx>();
    c->cap.set_layers(n_embd, layers);
    llama_context_params cp = llama_context_default_params();
    // embeddings mode: every token is an output, so the graph keeps all rows up to the exit layer
    // (otherwise the rows of non-output tokens are dropped before the last computed layer) and no LM
    // head is computed. The pooling is done by the capture callback, which accumulates over ubatches,
    // so llama.cpp's own pooling stays off and an input may span several ubatches: the compute
    // buffer is sized by n_ubatch, not by the longest input.
    cp.embeddings        = true;
    cp.pooling_type      = LLAMA_POOLING_TYPE_NONE;
    cp.n_ctx             = n_ctx;
    cp.n_batch           = n_ctx;
    cp.n_ubatch          = std::max(1, std::min(n_ubatch, n_ctx));
    // several items share one batch, one sequence each, in one unified KV cache of n_ctx cells
    cp.n_seq_max         = n_seq;
    cp.kv_unified        = true;
    cp.n_threads         = n_threads;
    cp.n_threads_batch   = n_threads;
    cp.cb_eval           = common_classifier_capture::cb_eval;
    cp.cb_eval_user_data = &c->cap;
    c->ctx = llama_init_from_model(model, cp);
    if (!c->ctx) {
        throw std::runtime_error("classifier: cannot create a context");
    }
    llama_set_n_layer_exit(c->ctx, exit);
    return c;
}

void server_classifier::init(llama_model * model, const std::vector<std::string> & paths, int32_t n_ctx, int32_t n_threads, bool features,
                             int32_t n_ubatch, bool options) {
    this->model    = model;
    this->n_embd   = llama_model_n_embd(model);
    this->n_ctx    = n_ctx;
    this->n_ubatch = n_ubatch;
    this->n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    this->n_layer_ = llama_model_n_layer(model);
    this->n_threads = n_threads;
    // hybrid and recurrent memories split a batch by sequence and may interleave the items' rows
    // (and a batch never holds more inputs than tokens; the context needs n_seq_max <= n_batch)
    this->n_seq_batch = (llama_model_is_recurrent(model) || llama_model_is_hybrid(model)) ? 1 : std::min(SERVER_CLASSIFIER_MAX_SEQ, n_ctx);

    for (const auto & path : paths) {
        common_classifier_head head;
        std::string err;
        if (!common_classifier_head_load(path, head, err)) {
            throw std::runtime_error("classifier: " + err);
        }
        if (head.n_embd != n_embd) {
            throw std::runtime_error("classifier: " + path + ": the head is for n_embd = " + std::to_string(head.n_embd) +
                                     ", the model has n_embd = " + std::to_string(n_embd));
        }
        if (head.layer < 1 || head.layer > n_layer_) {
            throw std::runtime_error("classifier: " + path + ": layer " + std::to_string(head.layer) +
                                     " is outside 1.." + std::to_string(n_layer_));
        }
        for (const auto & other : heads) {
            if (other.question_id == head.question_id) {
                throw std::runtime_error("classifier: " + path + ": question_id '" + head.question_id + "' is already used by " + other.path);
            }
        }
        if (!head.model.empty()) {
            const std::string here = common_classifier_model_desc(model);
            if (head.model != here) {
                fprintf(stderr, "classifier: warning: %s was fitted on '%s', this model is '%s'; its answers may be off\n",
                        path.c_str(), head.model.c_str(), here.c_str());
            }
        }
        heads.push_back(std::move(head));
    }

    std::vector<int32_t> layers;
    for (const auto & head : heads) {
        if (std::find(layers.begin(), layers.end(), head.layer) == layers.end()) {
            layers.push_back(head.layer);
        }
    }
    for (int32_t L : layers) {
        ctxs.push_back(make_ctx(model, n_ctx, n_ubatch, n_threads, n_embd, {L}, L, n_seq_batch));
    }
    if (options) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = n_ctx;
        cp.n_ubatch        = std::max(1, std::min(n_ubatch, n_ctx));
        cp.n_seq_max       = MAX_OPTIONS;
        cp.kv_unified      = true;
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        opt_ctx = llama_init_from_model(model, cp);
        if (!opt_ctx) {
            throw std::runtime_error("options: cannot create a context");
        }
    }
    if (features) {
        feat = make_ctx(model, n_ctx, n_ubatch, n_threads, n_embd, {}, 0, n_seq_batch);
    }
}

void server_classifier::check_tokens(const llama_tokens & tokens) const {
    if (tokens.empty()) {
        throw std::invalid_argument("classifier: empty input");
    }
    if ((int64_t) tokens.size() > n_ctx) {
        throw std::invalid_argument("classifier: input has " + std::to_string(tokens.size()) +
                                    " tokens, at most " + std::to_string(n_ctx) + " are allowed (--classifier-ctx)");
    }
    for (const llama_token t : tokens) {
        if (t < 0 || t >= n_vocab) {
            throw std::invalid_argument("classifier: token id " + std::to_string(t) + " is outside the vocabulary");
        }
    }
}

llama_tokens server_classifier::apply_template(const llama_tokens & tokens, int64_t & text_begin, int64_t & text_end) const {
    if (tmpl.empty() || tmpl == "none") {
        text_begin = 0;
        text_end   = (int64_t) tokens.size();
        return tokens;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tok = [&](const char * s) { return common_tokenize(vocab, s, false, false); };
    // the input's own leading BOS (if the tokenizer added one) stays first; the template goes after it
    llama_tokens out;
    size_t skip = 0;
    if (!tokens.empty() && llama_vocab_get_add_bos(vocab) && tokens[0] == llama_vocab_bos(vocab)) {
        out.push_back(tokens[0]);
        skip = 1;
    }
    const llama_tokens text(tokens.begin() + skip, tokens.end());
    auto add = [&](const llama_tokens & t) { out.insert(out.end(), t.begin(), t.end()); };
    if (tmpl == "echo") {
        add(tok("Rewrite the following paragraph: "));
        add(text);
        add(tok("\nThe rewritten paragraph: "));
        text_begin = (int64_t) out.size();
        add(text);
        text_end = (int64_t) out.size();
    } else if (tmpl == "prompteol") {
        add(tok("This sentence: \""));
        text_begin = (int64_t) out.size();
        add(text);
        text_end = (int64_t) out.size();
        add(tok("\" means in one word: \""));
    } else {
        throw std::invalid_argument("unknown template '" + tmpl + "' (echo, prompteol)");
    }
    if (text_end <= text_begin) {
        throw std::invalid_argument("template: the input has no text tokens");
    }
    return out;
}

void server_classifier::decode(server_classifier_ctx & c, const llama_tokens & tokens, bool keep_rows) {
    c.cap.reset();
    c.cap.keep_rows = keep_rows || std::any_of(heads.begin(), heads.end(), [](const common_classifier_head & h) {
        return h.pooling == COMMON_CLASSIFIER_POOL_SPAN; });
    llama_memory_clear(llama_get_memory(c.ctx), true);
    const bool bidir = bidirectional && &c == feat.get();
    if (bidir && (int32_t) tokens.size() > n_ubatch) {
        throw std::invalid_argument("bidirectional attention needs the whole input in one compute step: " +
                                    std::to_string(tokens.size()) + " tokens > --classifier-ubatch " + std::to_string(n_ubatch));
    }
    llama_set_causal_attn(c.ctx, !bidir);
    llama_tokens ids = tokens;
    const int32_t ret = llama_decode(c.ctx, llama_batch_get_one(ids.data(), (int32_t) ids.size()));
    llama_set_causal_attn(c.ctx, true);
    if (!c.cap.error.empty()) {
        throw std::runtime_error(c.cap.error);
    }
    if (ret != 0) {
        throw std::runtime_error("classifier: decode failed");
    }
    llama_synchronize(c.ctx);
    for (size_t i = 0; i < c.cap.layers.size(); i++) {
        if (c.cap.rows[i] != (int64_t) tokens.size()) {
            throw std::runtime_error("classifier: read " + std::to_string(c.cap.rows[i]) + " rows of layer " +
                                     std::to_string(c.cap.layers[i]) + " for " + std::to_string(tokens.size()) + " tokens");
        }
    }
}

static json server_classifier_answer_body(const common_classifier_head & head, const std::vector<double> & p);

json server_classifier_answer(const common_classifier_head & head, const std::vector<double> & p) {
    json j = server_classifier_answer_body(head, p);
    if (!head.version.empty()) {
        j["version"] = head.version;   // which fit produced the answer (cache keys, stale-head detection)
    }
    return j;
}

static json server_classifier_answer_body(const common_classifier_head & head, const std::vector<double> & p) {
    switch (head.type) {
        case COMMON_CLASSIFIER_NOUL:
            return json {{"type", "noul"}, {"noul", p[1]}};
        case COMMON_CLASSIFIER_CHOICE: {
            const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
            json probs = json::object();
            for (size_t k = 0; k < p.size(); k++) {
                probs[head.options[k]] = p[k];
            }
            return json {{"type", "choice"}, {"choice", head.options[best]}, {"confidence", p[best]}, {"probabilities", probs}};
        }
        case COMMON_CLASSIFIER_SCORE: {
            double expected = 0.0;
            json probs  = json::object();
            json legend = json::object();
            for (size_t k = 0; k < p.size(); k++) {
                expected += (double) k * p[k];
                probs[std::to_string(k)]  = p[k];
                legend[std::to_string(k)] = head.options[k];
            }
            // "score" is the expected level (a float); "level" is the most likely level (an integer) and "label"
            // its description, for callers that need a whole level
            const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
            return json {{"type", "score"}, {"score", expected}, {"level", (int) best}, {"label", head.options[best]},
                         {"confidence", p[best]}, {"legend", legend}, {"probabilities", probs}};
        }
    }
    return json();
}

// the pooled feature for one head and one item (rows [base, base + n) of the capture); span heads use the item's
// term and report how many span tokens they read (0 = no term or not found: mean1 instead)
static int32_t head_feature(const common_classifier_capture & cap, const llama_vocab * vocab, const common_classifier_head & head,
                            const llama_tokens & tokens, int64_t base, const std::string & term, bool batched, std::vector<float> & x) {
    if (head.pooling == COMMON_CLASSIFIER_POOL_SPAN && !term.empty()) {
        const auto pos = common_classifier_term_positions(vocab, tokens, term, head.span_window);
        if (!pos.empty() && cap.get_positions(head.layer, base, pos, x)) {
            return (int32_t) pos.size();
        }
    }
    if (batched) {
        cap.get_range(head.layer, head.pooling, base, base + (int64_t) tokens.size(), x);
    } else {
        cap.get(head.layer, head.pooling, x);
    }
    return 0;
}

json server_classifier::classify(const llama_tokens & tokens, const std::string & term) {
    check_tokens(tokens);
    json answers = json::object();
    std::vector<float> x;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (auto & c : ctxs) {
        decode(*c, tokens);
        for (const auto & head : heads) {
            if (head.layer != c->cap.layers[0]) {
                continue;
            }
            const int32_t n_span = head_feature(c->cap, vocab, head, tokens, 0, term, false, x);
            answers[head.question_id] = server_classifier_answer(head, common_classifier_probs(head, x.data()));
            if (head.pooling == COMMON_CLASSIFIER_POOL_SPAN) answers[head.question_id]["span_tokens"] = n_span;
        }
    }
    return answers;
}

void server_classifier::decode_batch(server_classifier_ctx & c, const std::vector<llama_tokens> & items, int64_t n_total) {
    c.cap.reset();
    c.cap.keep_rows = true;
    llama_memory_clear(llama_get_memory(c.ctx), true);
    const bool bidir = bidirectional && &c == feat.get();
    llama_set_causal_attn(c.ctx, !bidir);   // features_batch only batches what fits in one ubatch when bidirectional
    llama_batch batch = llama_batch_init((int32_t) n_total, 0, 1);
    for (size_t s = 0; s < items.size(); s++) {
        for (size_t i = 0; i < items[s].size(); i++) {
            const int32_t n = batch.n_tokens++;
            batch.token[n]     = items[s][i];
            batch.pos[n]       = (llama_pos) i;
            batch.n_seq_id[n]  = 1;
            batch.seq_id[n][0] = (llama_seq_id) s;
            batch.logits[n]    = 1;
        }
    }
    const int32_t ret = llama_decode(c.ctx, batch);
    llama_set_causal_attn(c.ctx, true);
    llama_batch_free(batch);
    c.cap.keep_rows = false;
    if (!c.cap.error.empty()) {
        throw std::runtime_error(c.cap.error);
    }
    if (ret != 0) {
        throw std::runtime_error("classifier: decode failed");
    }
    llama_synchronize(c.ctx);
    for (size_t i = 0; i < c.cap.layers.size(); i++) {
        if (c.cap.rows[i] != n_total) {
            throw std::runtime_error("classifier: read " + std::to_string(c.cap.rows[i]) + " rows of layer " +
                                     std::to_string(c.cap.layers[i]) + " for " + std::to_string(n_total) + " tokens");
        }
    }
}

std::vector<json> server_classifier::classify_batch(const std::vector<llama_tokens> & items, const std::vector<std::string> & terms) {
    int64_t n_total = 0;
    for (const auto & tokens : items) {
        check_tokens(tokens);
        n_total += (int64_t) tokens.size();
    }
    std::vector<json> out;
    if (items.size() == 1 || (int32_t) items.size() > n_seq_batch || n_total > n_ctx) {
        for (size_t k = 0; k < items.size(); k++) {
            out.push_back(classify(items[k], k < terms.size() ? terms[k] : std::string()));
        }
        return out;
    }
    out.assign(items.size(), json::object());
    std::vector<float> x;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (auto & c : ctxs) {
        decode_batch(*c, items, n_total);
        int64_t begin = 0;
        for (size_t k = 0; k < items.size(); k++) {
            const int64_t end = begin + (int64_t) items[k].size();
            for (const auto & head : heads) {
                if (head.layer != c->cap.layers[0]) {
                    continue;
                }
                const int32_t n_span = head_feature(c->cap, vocab, head, items[k], begin, k < terms.size() ? terms[k] : std::string(), true, x);
                out[k][head.question_id] = server_classifier_answer(head, common_classifier_probs(head, x.data()));
                if (head.pooling == COMMON_CLASSIFIER_POOL_SPAN) out[k][head.question_id]["span_tokens"] = n_span;
            }
            begin = end;
        }
    }
    return out;
}

std::vector<float> server_classifier::features_raw(const llama_tokens & tokens, const std::vector<int32_t> & layers,
                                                  const std::vector<common_classifier_pooling> & poolings,
                                                  const std::string & term, int32_t span_window) {
    if (!feat) {
        throw std::invalid_argument("features are disabled, start the server with --features");
    }
    check_tokens(tokens);
    if (layers.empty() || poolings.empty()) {
        throw std::invalid_argument("features: no layers or poolings requested");
    }
    for (int32_t L : layers) {
        if (L < 1 || L > n_layer_) {
            throw std::invalid_argument("features: layer " + std::to_string(L) + " is outside 1.." + std::to_string(n_layer_));
        }
    }
    feat->cap.set_layers(n_embd, layers);
    // compute only as deep as the deepest requested layer (the memory is cleared for every input anyway)
    llama_set_n_layer_exit(feat->ctx, *std::max_element(layers.begin(), layers.end()));
    const bool span = std::find(poolings.begin(), poolings.end(), COMMON_CLASSIFIER_POOL_SPAN) != poolings.end();
    if (!tmpl.empty() && tmpl != "none") {
        if (span) {
            throw std::invalid_argument("features: span pooling cannot be combined with a template");
        }
        int64_t tb = 0, te = 0;
        const llama_tokens t2 = apply_template(tokens, tb, te);
        decode(*feat, t2, true);
        std::vector<float> out, x;
        for (int32_t L : layers) {
            for (auto pool : poolings) {
                // last: the final token of the templated input; mean / mean1: the mean over the text range
                if (pool == COMMON_CLASSIFIER_POOL_LAST) {
                    feat->cap.get_range(L, pool, 0, (int64_t) t2.size(), x);
                } else {
                    feat->cap.get_range(L, COMMON_CLASSIFIER_POOL_MEAN, tb, te, x);
                }
                out.insert(out.end(), x.begin(), x.end());
            }
        }
        return out;
    }
    decode(*feat, tokens, span);
    std::vector<int32_t> pos;
    if (span && !term.empty()) {
        pos = common_classifier_term_positions(llama_model_get_vocab(model), tokens, term, span_window);
    }
    std::vector<float> out;
    out.reserve(layers.size() * poolings.size() * n_embd);
    std::vector<float> x;
    for (int32_t L : layers) {
        for (auto pool : poolings) {
            // span without a term found: mean1 (the capture's fallback for span)
            if (!(pool == COMMON_CLASSIFIER_POOL_SPAN && !pos.empty() && feat->cap.get_positions(L, 0, pos, x))) {
                feat->cap.get(L, pool, x);
            }
            out.insert(out.end(), x.begin(), x.end());
        }
    }
    return out;
}

void server_classifier::add_head(const common_classifier_head & head_in) {
    common_classifier_head head = head_in;
    if (head.model.empty() && model) {
        head.model = common_classifier_model_desc(model);   // live-trained heads: the model they were fitted on
    }
    const std::string err = head.validate();
    if (!err.empty()) {
        throw std::invalid_argument("classifier: " + err);
    }
    if (head.n_embd != n_embd || head.layer < 1 || head.layer > n_layer_) {
        throw std::invalid_argument("classifier: the head does not fit the model");
    }
    bool have_ctx = false;
    for (const auto & c : ctxs) {
        have_ctx = have_ctx || c->cap.layers[0] == head.layer;
    }
    if (!have_ctx) {
        ctxs.push_back(make_ctx(model, n_ctx, n_ubatch, n_threads, n_embd, {head.layer}, head.layer, n_seq_batch));
    }
    bool replaced = false;
    for (auto & h : heads) {
        if (h.question_id == head.question_id) {
            h = head;
            replaced = true;
        }
    }
    if (!replaced) {
        heads.push_back(head);
    }
    // a replaced head may have left its old layer without heads
    ctxs.erase(std::remove_if(ctxs.begin(), ctxs.end(), [&](const std::unique_ptr<server_classifier_ctx> & c) {
        return std::none_of(heads.begin(), heads.end(), [&](const common_classifier_head & h) { return h.layer == c->cap.layers[0]; });
    }), ctxs.end());
}

json server_classifier::features(const llama_tokens & tokens, const std::vector<int32_t> & layers,
                                 const std::vector<common_classifier_pooling> & poolings) {
    const std::vector<float> raw = features_raw(tokens, layers, poolings);
    json out = json::object();
    size_t off = 0;
    for (int32_t L : layers) {
        json per = json::object();
        for (auto pool : poolings) {
            per[common_classifier_pooling_name(pool)] = std::vector<float>(raw.begin() + off, raw.begin() + off + n_embd);
            off += n_embd;
        }
        out[std::to_string(L)] = per;
    }
    return out;
}

std::vector<json> server_classifier::features_batch(const std::vector<llama_tokens> & items_in, const std::vector<int32_t> & layers,
                                                   const std::vector<common_classifier_pooling> & poolings) {
    // with a template: the templated inputs, and per input the text range that mean pooling reads
    const bool templated = !tmpl.empty() && tmpl != "none";
    std::vector<llama_tokens> items;
    std::vector<std::pair<int64_t, int64_t>> ranges;
    for (const auto & tokens : items_in) {
        check_tokens(tokens);
        int64_t tb = 0, te = 0;
        items.push_back(apply_template(tokens, tb, te));
        ranges.emplace_back(tb, te);
    }
    if (templated && std::find(poolings.begin(), poolings.end(), COMMON_CLASSIFIER_POOL_SPAN) != poolings.end()) {
        throw std::invalid_argument("features: span pooling cannot be combined with a template");
    }
    int64_t n_total = 0;
    for (const auto & tokens : items) {
        n_total += (int64_t) tokens.size();
    }
    std::vector<json> out;
    if (!feat || items.size() == 1 || (int32_t) items.size() > n_seq_batch || n_total > n_ctx ||
        (bidirectional && n_total > n_ubatch)) {
        for (const auto & tokens : items_in) {   // features() applies the template itself
            out.push_back(features(tokens, layers, poolings));
        }
        return out;
    }
    if (layers.empty() || poolings.empty()) {
        throw std::invalid_argument("features: no layers or poolings requested");
    }
    for (int32_t L : layers) {
        if (L < 1 || L > n_layer_) {
            throw std::invalid_argument("features: layer " + std::to_string(L) + " is outside 1.." + std::to_string(n_layer_));
        }
    }
    feat->cap.set_layers(n_embd, layers);
    llama_set_n_layer_exit(feat->ctx, *std::max_element(layers.begin(), layers.end()));
    decode_batch(*feat, items, n_total);
    std::vector<float> x;
    int64_t begin = 0;
    for (size_t k = 0; k < items.size(); k++) {
        const int64_t end = begin + (int64_t) items[k].size();
        json o = json::object();
        for (int32_t L : layers) {
            json per = json::object();
            for (auto pool : poolings) {
                if (templated && pool != COMMON_CLASSIFIER_POOL_LAST) {
                    feat->cap.get_range(L, COMMON_CLASSIFIER_POOL_MEAN, begin + ranges[k].first, begin + ranges[k].second, x);
                } else {
                    feat->cap.get_range(L, pool, begin, end, x);
                }
                per[common_classifier_pooling_name(pool)] = x;
            }
            o[std::to_string(L)] = per;
        }
        out.push_back(o);
        begin = end;
    }
    return out;
}

std::vector<json> server_classifier::classify_questions(const std::vector<llama_tokens> & items, const std::vector<question> & questions,
                                                      int32_t layer, common_classifier_pooling pooling, double temperature,
                                                      const std::vector<llama_tokens> & unlabeled, bool unlabeled_inputs,
                                                      bool idf) {
    // layer 0: averaged input embeddings, no forward pass
    const bool emb = layer == 0;
    if (!emb && !feat) {
        throw std::invalid_argument("questions in the request need the features context, start the server with --features "
                                    "(or use \"layer\": 0, which needs no context)");
    }
    if (layer < 0 || layer > n_layer_) {
        throw std::invalid_argument("questions: layer " + std::to_string(layer) + " is outside 0.." + std::to_string(n_layer_));
    }
    if (!(temperature > 0)) {
        throw std::invalid_argument("questions: the temperature must be positive");
    }
    const std::vector<int32_t> layers = {layer};
    const std::vector<common_classifier_pooling> pools = {pooling};
    const std::string pname = emb ? (idf ? "idf" : "mean") : common_classifier_pooling_name(pooling);
    // fast mode: token weights from the document frequencies over all of the request's texts
    std::unordered_map<llama_token, double> wt;
    if (emb && idf) {
        std::unordered_map<llama_token, int> df;
        size_t ndoc = 0;
        auto count = [&](const llama_tokens & t) {
            for (llama_token x : std::unordered_set<llama_token>(t.begin(), t.end())) df[x]++;
            ndoc++;
        };
        for (const auto & t : items) count(t);
        for (const auto & q : questions) for (const auto & d : q.descriptions) for (const auto & t : d) count(t);
        for (const auto & t : unlabeled) count(t);
        for (const auto & [x, n] : df) wt[x] = std::log((double) ndoc / (double) n);
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<float> row(n_embd);
    auto embed_pool = [&](const llama_tokens & t) {
        std::vector<double> acc(n_embd, 0.0);
        double wsum = 0.0;
        for (int pass = 0; pass < 2 && wsum == 0.0; pass++) {
            // the second pass (plain mean) only when every token had weight 0, e.g. a text that is in every document
            for (llama_token x : t) {
                if (llama_vocab_is_control(vocab, x)) continue;
                const double w = (pass == 0 && idf) ? wt[x] : 1.0;
                if (w == 0.0) continue;
                if (llama_model_token_embd_row(model, x, row.data(), n_embd) != n_embd) {
                    throw std::runtime_error("questions: layer 0 needs input embedding rows of width n_embd for this model");
                }
                for (int32_t j = 0; j < n_embd; j++) acc[j] += w * row[j];
                wsum += w;
            }
        }
        std::vector<float> v(n_embd, 0.0f);
        if (wsum > 0.0) for (int32_t j = 0; j < n_embd; j++) v[j] = (float) (acc[j] / wsum);
        return v;
    };
    // the inputs' pooled vectors
    std::vector<json> fx;
    if (emb) {
        for (const auto & t : items) fx.push_back(json {{"0", {{pname, embed_pool(t)}}}});
    } else {
        fx = features_batch(items, layers, pools);
    }
    // the descriptions' pooled vectors, cached by layer, pooling and tokens (in the fast mode only within the request,
    // since the weights depend on its texts)
    std::map<std::string, std::vector<float>> local_cache;
    auto describe = [&](const llama_tokens & t) -> const std::vector<float> & {
        std::string key = std::to_string(layer) + "|" + pname + "|";
        for (llama_token x : t) key += std::to_string(x) + ",";
        auto & cache = emb ? local_cache : desc_cache;
        auto it = cache.find(key);
        if (it == cache.end()) {
            const std::vector<float> v = emb ? embed_pool(t) : features_raw(t, layers, pools);
            if (cache.size() > 4096) cache.clear();
            it = cache.emplace(key, v).first;
        }
        return it->second;
    };
    // the top principal direction of the unlabeled population (power iteration on the centred pooled vectors)
    std::vector<double> u;
    size_t n_unlabeled = 0;
    if (unlabeled_inputs || !unlabeled.empty()) {
        std::vector<std::vector<double>> U;
        if (unlabeled_inputs) {
            for (const auto & f : fx) {
                const std::vector<float> x = f[std::to_string(layer)][pname].get<std::vector<float>>();
                U.emplace_back(x.begin(), x.end());
            }
        } else {
            for (const auto & t : unlabeled) {
                const auto & v = describe(t);
                U.emplace_back(v.begin(), v.end());
            }
        }
        n_unlabeled = U.size();
        if (n_unlabeled < 2) {
            throw std::invalid_argument("questions: \"unlabeled\" needs 2 or more texts (50-100 work well)");
        }
        std::vector<double> mean(n_embd, 0.0);
        for (const auto & r : U) for (int32_t j = 0; j < n_embd; j++) mean[j] += r[j] / (double) n_unlabeled;
        for (auto & r : U) for (int32_t j = 0; j < n_embd; j++) r[j] -= mean[j];
        u.assign(n_embd, 0.0);
        for (int32_t j = 0; j < n_embd; j++) u[j] = 1.0 + 1e-3 * (double) (j % 7);   // deterministic start
        for (int it = 0; it < 100; it++) {
            std::vector<double> nu(n_embd, 0.0);
            for (const auto & r : U) {
                double d = 0.0;
                for (int32_t j = 0; j < n_embd; j++) d += r[j] * u[j];
                for (int32_t j = 0; j < n_embd; j++) nu[j] += d * r[j];
            }
            double nn = 0.0;
            for (double v : nu) nn += v * v;
            nn = std::sqrt(nn);
            if (nn == 0.0) { u.clear(); break; }
            double delta = 0.0;
            for (int32_t j = 0; j < n_embd; j++) { nu[j] /= nn; delta += std::fabs(nu[j] - u[j]); }
            u.swap(nu);
            if (delta < 1e-9) break;
        }
    }
    auto drop_u = [&](std::vector<double> & v) {
        if (u.empty()) return;
        double d = 0.0;
        for (int32_t j = 0; j < n_embd; j++) d += v[j] * u[j];
        for (int32_t j = 0; j < n_embd; j++) v[j] -= d * u[j];
    };
    std::vector<json> out(items.size(), json::object());
    for (const auto & q : questions) {
        const size_t K = q.descriptions.size();
        std::vector<std::vector<double>> P(K, std::vector<double>(n_embd));
        std::vector<double> c(n_embd, 0.0);
        for (size_t k = 0; k < K; k++) {
            // an answer's prototype: the mean of the pooled vectors of its description or examples
            const double w = 1.0 / (double) q.descriptions[k].size();
            for (const auto & t : q.descriptions[k]) {
                const auto & v = describe(t);
                for (int32_t j = 0; j < n_embd; j++) P[k][j] += w * v[j];
            }
            for (int32_t j = 0; j < n_embd; j++) c[j] += P[k][j] / (double) K;
        }
        for (auto & p : P) {
            for (int32_t j = 0; j < n_embd; j++) p[j] -= c[j];
            double n0 = 0.0, n1 = 0.0;
            for (double v : p) n0 += v * v;
            drop_u(p);
            for (double v : p) n1 += v * v;
            // a prototype that lay along the removed direction is gone: zero it rather than normalise rounding noise
            if (n1 <= 1e-12 * n0) std::fill(p.begin(), p.end(), 0.0);
        }
        for (size_t i = 0; i < items.size(); i++) {
            const std::vector<float> x = fx[i][std::to_string(layer)][pname].get<std::vector<float>>();
            std::vector<double> xc(n_embd);
            for (int32_t j = 0; j < n_embd; j++) xc[j] = x[j] - c[j];
            drop_u(xc);
            double xn = 0.0;
            for (int32_t j = 0; j < n_embd; j++) xn += xc[j] * xc[j];
            xn = std::sqrt(xn) + 1e-12;
            std::vector<double> cs(K);
            for (size_t k = 0; k < K; k++) {
                double d = 0.0, pn = 0.0;
                for (int32_t j = 0; j < n_embd; j++) { d += xc[j] * P[k][j]; pn += P[k][j] * P[k][j]; }
                cs[k] = d / (xn * (std::sqrt(pn) + 1e-12));
            }
            const double mx = *std::max_element(cs.begin(), cs.end());
            double z = 0.0;
            for (double v : cs) z += std::exp((v - mx) / temperature);
            json sims = json::object(), probs = json::object();
            size_t best = 0;
            for (size_t k = 0; k < K; k++) {
                sims[q.names[k]]  = cs[k];
                probs[q.names[k]] = std::exp((cs[k] - mx) / temperature) / z;
                if (cs[k] > cs[best]) best = k;
            }
            json a;
            if (q.type == "noul") {
                a = {{"type", "noul"}, {"noul", probs["true"]}, {"similarity", sims}};
            } else {
                a = {{"type", "choice"}, {"choice", q.names[best]}, {"confidence", probs[q.names[best]]},
                     {"probabilities", probs}, {"similarity", sims}};
            }
            a["method"] = "similarity";
            if (!u.empty()) a["unlabeled"] = n_unlabeled;
            out[i][q.id] = a;
        }
    }
    return out;
}

// log-softmax of one row of logits at the token t
static double token_logprob(const float * logits, int32_t n_vocab, llama_token t) {
    float mx = logits[0];
    for (int32_t i = 1; i < n_vocab; i++) mx = std::max(mx, logits[i]);
    double z = 0.0;
    for (int32_t i = 0; i < n_vocab; i++) z += std::exp((double) logits[i] - mx);
    return (double) logits[t] - mx - std::log(z);
}

json server_classifier::score_options(const std::vector<llama_tokens> & full, const std::vector<std::string> & names) {
    if (!opt_ctx) {
        throw std::invalid_argument("options are disabled, start the server with --options");
    }
    const size_t K = full.size();
    if (K < 2 || (int32_t) K > MAX_OPTIONS) {
        throw std::invalid_argument("options: give 2.." + std::to_string(MAX_OPTIONS) + " options");
    }
    // shared prefix: the common prefix of all prompt + option sequences, leaving every option one own token
    size_t p = full[0].size();
    for (size_t k = 1; k < K; k++) {
        size_t q = 0;
        while (q < p && q < full[k].size() && full[k][q] == full[0][q]) q++;
        p = q;
    }
    for (const auto & f : full) {
        check_tokens(f);
        p = std::min(p, f.size() - 1);
    }
    if (p == 0) {
        throw std::invalid_argument("options: the prompt is empty");
    }
    int64_t n_total = (int64_t) p;
    for (const auto & f : full) n_total += (int64_t) (f.size() - p);
    if (n_total > n_ctx) {
        throw std::invalid_argument("options: prompt and options have " + std::to_string(n_total) + " tokens, at most " +
                                    std::to_string(n_ctx) + " are allowed (--classifier-ctx)");
    }
    llama_memory_t mem = llama_get_memory(opt_ctx);
    llama_memory_clear(mem, true);
    // 1. the shared prefix in sequence 0, logits of its last token only
    llama_batch b = llama_batch_init((int32_t) n_total, 0, 1);
    for (size_t i = 0; i < p; i++) {
        b.token[b.n_tokens] = full[0][i]; b.pos[b.n_tokens] = (llama_pos) i; b.n_seq_id[b.n_tokens] = 1;
        b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = i + 1 == p; b.n_tokens++;
    }
    if (llama_decode(opt_ctx, b) != 0) { llama_batch_free(b); throw std::runtime_error("options: decode failed"); }
    std::vector<float> last(llama_get_logits_ith(opt_ctx, (int32_t) p - 1), llama_get_logits_ith(opt_ctx, (int32_t) p - 1) + n_vocab);
    // 2. every option's own tokens but its last, one sequence each, after copying the prefix
    for (size_t k = 1; k < K; k++) {
        llama_memory_seq_cp(mem, 0, (llama_seq_id) k, -1, -1);
    }
    b.n_tokens = 0;
    std::vector<std::vector<int32_t>> idx(K);
    for (size_t k = 0; k < K; k++) {
        for (size_t i = p; i + 1 < full[k].size(); i++) {
            idx[k].push_back(b.n_tokens);
            b.token[b.n_tokens] = full[k][i]; b.pos[b.n_tokens] = (llama_pos) i; b.n_seq_id[b.n_tokens] = 1;
            b.seq_id[b.n_tokens][0] = (llama_seq_id) k; b.logits[b.n_tokens] = 1; b.n_tokens++;
        }
    }
    if (b.n_tokens > 0 && llama_decode(opt_ctx, b) != 0) { llama_batch_free(b); throw std::runtime_error("options: decode failed"); }
    llama_batch_free(b);
    // 3. summed log-probs
    std::vector<double> lp(K, 0.0);
    for (size_t k = 0; k < K; k++) {
        lp[k] = token_logprob(last.data(), n_vocab, full[k][p]);
        for (size_t j = 0; j < idx[k].size(); j++) {
            lp[k] += token_logprob(llama_get_logits_ith(opt_ctx, idx[k][j]), n_vocab, full[k][p + 1 + j]);
        }
    }
    const double mx = *std::max_element(lp.begin(), lp.end());
    double z = 0.0;
    for (double v : lp) z += std::exp(v - mx);
    json opts = json::array();
    json probs = json::object();
    size_t best = 0;
    for (size_t k = 0; k < K; k++) {
        const double pk = std::exp(lp[k] - mx) / z;
        opts.push_back({{"option", names[k]}, {"logprob", lp[k]}, {"n_tokens", full[k].size() - p}, {"probability", pk}});
        probs[names[k]] = pk;
        if (lp[k] > lp[best]) best = k;
    }
    return json {{"options", opts}, {"probabilities", probs}, {"best", names[best]}, {"confidence", std::exp(lp[best] - mx) / z},
                 {"prompt_tokens", p}};
}

json server_classifier::info() const {
    json out = json::array();
    for (const auto & head : heads) {
        json h = {
            {"path",        head.path},
            {"question_id", head.question_id},
            {"type",        common_classifier_type_name(head.type)},
            {"pooling",     common_classifier_pooling_name(head.pooling)},
            {"layer",       head.layer},
        };
        if (!head.version.empty()) h["version"] = head.version;
        if (!head.model.empty())   h["model"]   = head.model;
        if (!head.options.empty()) {
            h["options"] = head.options;
        }
        out.push_back(h);
    }
    return out;
}
