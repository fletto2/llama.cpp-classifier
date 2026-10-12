// llama-kv-fold: calibrate a KV-cache fold and write it into a copy of the model.
//
// For each attention layer, key head h and channel c it computes
//   shift[c,h] = mean of the key after RoPE
//   fold[c,h]  = rms_k^alpha / rms_q^(1 - alpha), averaged within each RoPE pair and normalised to a geometric mean of 1
// over a calibration text (the first token of each chunk, the attention sink, is left out). With these tensors in the
// GGUF (blk.N.attn_k_fold / blk.N.attn_k_shift), llama.cpp stores keys as (k - shift) / fold and scales the queries of
// that key head by fold. Attention is unchanged up to rounding (q.shift is the same for every key of a query and
// softmax removes it, and the fold is equal within each RoPE pair, so it commutes with RoPE); the cached keys have
// evened-out channels and lose much less to a quantized cache type (-ctk q4_0 / iq4_nl).
//
// usage: llama-kv-fold -m model.gguf -f calib.txt -o out.gguf [-c 512] [--chunks 64] [--alpha 0.5] [--no-shift] [-t N]

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <array>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct layer_stats {
    int64_t hd = 0, nkh = 0, nh = 0;
    std::vector<double> k1, k2, q2;   // [nkh * hd]
    double nk = 0, nq = 0;
};

struct calib_state {
    std::map<int, std::vector<float>> qbuf, kbuf;   // last "Qcur-L" / "Kcur-L" of the current graph
    std::map<int, std::array<int64_t, 3>> qshape, kshape;
};

static bool parse_layer(const char * name, const char * prefix, int & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return false;
    }
    char * end = nullptr;
    il = (int) strtol(name + n, &end, 10);
    return end != name + n && *end == '\0';
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (calib_state *) user_data;
    int il = -1;
    const bool is_q = parse_layer(t->name, "Qcur-", il);
    const bool is_k = !is_q && parse_layer(t->name, "Kcur-", il);
    if (!is_q && !is_k) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || ggml_n_dims(t) > 3) {
        return true;
    }
    std::vector<float> tmp(ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
    } else {   // strided view: fetch the span and gather
        std::vector<uint8_t> raw(t->nb[2] * (t->ne[2] - 1) + t->nb[1] * (t->ne[1] - 1) + t->nb[0] * t->ne[0]);
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        size_t i = 0;
        for (int64_t c = 0; c < t->ne[2]; c++)
            for (int64_t b = 0; b < t->ne[1]; b++)
                for (int64_t a = 0; a < t->ne[0]; a++)
                    tmp[i++] = *(const float *) (raw.data() + c * t->nb[2] + b * t->nb[1] + a * t->nb[0]);
    }
    // a layer names several tensors Qcur / Kcur (before and after norm and RoPE); the last one is the attention input
    (is_q ? st->qbuf : st->kbuf)[il] = std::move(tmp);
    (is_q ? st->qshape : st->kshape)[il] = { t->ne[0], t->ne[1], t->ne[2] };
    return true;
}

int main(int argc, char ** argv) {
    std::string fmodel, ftext, fout;
    int n_ctx = 512, n_chunks = 64, n_threads = 8;
    float alpha = 0.5f;
    bool use_shift = true;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); } return std::string(argv[++i]); };
        if      (a == "-m") fmodel = next();
        else if (a == "-f") ftext = next();
        else if (a == "-o") fout = next();
        else if (a == "-c") n_ctx = std::stoi(next());
        else if (a == "--chunks") n_chunks = std::stoi(next());
        else if (a == "--alpha") alpha = std::stof(next());
        else if (a == "--no-shift") use_shift = false;
        else if (a == "-t") n_threads = std::stoi(next());
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (fmodel.empty() || ftext.empty() || fout.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f calib.txt -o out.gguf [-c 512] [--chunks 64] [--alpha 0.5] [--no-shift] [-t N]\n", argv[0]);
        return 1;
    }

    // refuse a model that already has a fold: its Qcur / Kcur would be calibrated before the old fold
    {
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * g = gguf_init_from_file(fmodel.c_str(), gp);
        if (!g) { fprintf(stderr, "cannot read %s\n", fmodel.c_str()); return 1; }
        for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
            const char * n = gguf_get_tensor_name(g, i);
            if (strstr(n, ".attn_k_fold.") || strstr(n, ".attn_k_shift.")) {
                fprintf(stderr, "%s already has a KV-cache fold (%s); calibrate the original model\n", fmodel.c_str(), n);
                return 1;
            }
        }
        gguf_free(g);
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(fmodel.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", fmodel.c_str()); return 1; }

    calib_state st;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create the context\n"); return 1; }

    std::ifstream in(ftext);
    std::stringstream ss; ss << in.rdbuf();
    const std::string text = ss.str();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < 0) { toks.resize(-nt); nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false); }
    toks.resize(nt);

    const llama_rope_type rope = llama_model_rope_type(model);
    std::map<int, layer_stats> stats;
    int done = 0;
    for (int c = 0; c < n_chunks && (size_t) (c + 1) * n_ctx <= toks.size(); c++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        st.qbuf.clear(); st.kbuf.clear();
        std::vector<llama_token> chunk(toks.begin() + (size_t) c * n_ctx, toks.begin() + (size_t) (c + 1) * n_ctx);
        if (llama_decode(ctx, llama_batch_get_one(chunk.data(), (int) chunk.size()))) {
            fprintf(stderr, "decode failed on chunk %d\n", c);
            return 1;
        }
        for (auto & [il, kv] : st.kbuf) {
            if (!st.qbuf.count(il)) continue;
            const auto ks = st.kshape[il], qs = st.qshape[il];
            const int64_t hd = ks[0], nkh = ks[1], T = ks[2], nh = qs[1];
            if (qs[0] != hd || qs[2] != T || nh % nkh) continue;
            auto & s = stats[il];
            if (s.hd == 0) { s.hd = hd; s.nkh = nkh; s.nh = nh; s.k1.assign(hd * nkh, 0); s.k2.assign(hd * nkh, 0); s.q2.assign(hd * nkh, 0); }
            const auto & qv = st.qbuf[il];
            const int64_t g = nh / nkh;
            for (int64_t t = 1; t < T; t++) {   // position 0 (the sink) left out
                for (int64_t h = 0; h < nkh; h++) {
                    for (int64_t d = 0; d < hd; d++) {
                        const double x = kv[(t * nkh + h) * hd + d];
                        s.k1[h * hd + d] += x; s.k2[h * hd + d] += x * x;
                        for (int64_t j = 0; j < g; j++) {
                            const double y = qv[(t * nh + h * g + j) * hd + d];
                            s.q2[h * hd + d] += y * y;
                        }
                    }
                }
            }
            s.nk += T - 1; s.nq += (T - 1) * g;
        }
        done++;
        fprintf(stderr, "\rchunk %d / %d", done, n_chunks);
    }
    fprintf(stderr, "\n");
    if (done == 0 || stats.empty()) { fprintf(stderr, "no calibration data (text too short, or no Qcur/Kcur tensors)\n"); return 1; }

    // fold and shift tensors
    const size_t mem = stats.size() * 2 * (ggml_tensor_overhead() + stats.begin()->second.hd * stats.begin()->second.nkh * 4 * 4) + 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * gctx = ggml_init(ip);
    std::vector<ggml_tensor *> added;
    for (auto & [il, s] : stats) {
        const int64_t hd = s.hd, nkh = s.nkh;
        ggml_tensor * tf = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hd, nkh);
        ggml_tensor * tc = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hd, nkh);
        ggml_format_name(tf, "blk.%d.attn_k_fold.weight", il);
        ggml_format_name(tc, "blk.%d.attn_k_shift.weight", il);
        float * f = (float *) tf->data, * cs = (float *) tc->data;
        for (int64_t h = 0; h < nkh; h++) {
            std::vector<double> rk(hd), rq(hd), sc(hd, 1.0);
            for (int64_t d = 0; d < hd; d++) {
                rk[d] = std::sqrt(s.k2[h * hd + d] / s.nk);
                rq[d] = std::sqrt(s.q2[h * hd + d] / s.nq);
            }
            // pairs RoPE rotates together must share the fold
            auto partner = [&](int64_t d) -> int64_t {
                if (rope == LLAMA_ROPE_TYPE_NONE) return d;
                if (rope == LLAMA_ROPE_TYPE_NORM) return d ^ 1;
                return d < hd / 2 ? d + hd / 2 : d - hd / 2;
            };
            double lsum = 0;
            for (int64_t d = 0; d < hd; d++) {
                const int64_t e = partner(d);
                const double pk = std::sqrt((rk[d] * rk[d] + rk[e] * rk[e]) / 2);
                const double pq = std::sqrt((rq[d] * rq[d] + rq[e] * rq[e]) / 2);
                sc[d] = (pk > 0 && pq > 0) ? std::pow(pk, alpha) / std::pow(pq, 1 - alpha) : 1.0;
                lsum += std::log(sc[d]);
            }
            const double gm = std::exp(lsum / hd);
            for (int64_t d = 0; d < hd; d++) {
                f[h * hd + d]  = (float) (sc[d] / gm);
                cs[h * hd + d] = use_shift ? (float) (s.k1[h * hd + d] / s.nk) : 0.0f;
            }
        }
        added.push_back(tf);
        if (use_shift) added.push_back(tc);
    }

    // write the copy: metadata + original tensors + the fold tensors
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    gguf_context * gin = gguf_init_from_file(fmodel.c_str(), gp);
    gguf_context * gout = gguf_init_empty();
    gguf_set_kv(gout, gin);
    gguf_set_val_f32(gout, "kv_fold.alpha", alpha);
    gguf_set_val_u32(gout, "kv_fold.tokens", (uint32_t) (done * n_ctx));
    const int64_t n_in = gguf_get_n_tensors(gin);
    for (int64_t i = 0; i < n_in; i++) {
        gguf_add_tensor(gout, ggml_get_tensor(meta_ctx, gguf_get_tensor_name(gin, i)));
    }
    for (ggml_tensor * t : added) {
        gguf_add_tensor(gout, t);
    }
    FILE * fo = fopen(fout.c_str(), "wb");
    FILE * fi = fopen(fmodel.c_str(), "rb");
    if (!fo || !fi) { fprintf(stderr, "cannot open %s or %s\n", fout.c_str(), fmodel.c_str()); return 1; }
    std::vector<uint8_t> buf(gguf_get_meta_size(gout));
    gguf_get_meta_data(gout, buf.data());
    fwrite(buf.data(), 1, buf.size(), fo);
    const size_t align = gguf_get_alignment(gout);
    std::vector<uint8_t> io(64 << 20);
    auto pad = [&](size_t n) { static const uint8_t z[256] = {0}; size_t p = GGML_PAD(n, align) - n; while (p) { size_t w = p < sizeof(z) ? p : sizeof(z); fwrite(z, 1, w, fo); p -= w; } };
    for (int64_t i = 0; i < n_in; i++) {
        const ggml_tensor * t = ggml_get_tensor(meta_ctx, gguf_get_tensor_name(gin, i));
        size_t n = ggml_nbytes(t);
        fseeko(fi, (off_t) (gguf_get_data_offset(gin) + gguf_get_tensor_offset(gin, i)), SEEK_SET);
        const size_t total = n;
        while (n) {
            const size_t r = n < io.size() ? n : io.size();
            if (fread(io.data(), 1, r, fi) != r) { fprintf(stderr, "short read in %s\n", fmodel.c_str()); return 1; }
            fwrite(io.data(), 1, r, fo);
            n -= r;
        }
        pad(total);
    }
    for (ggml_tensor * t : added) {
        fwrite(t->data, 1, ggml_nbytes(t), fo);
        pad(ggml_nbytes(t));
    }
    fclose(fi);
    fclose(fo);
    fprintf(stderr, "%s: fold%s for %zu layers from %d tokens (alpha %.2f)\n", fout.c_str(), use_shift ? " + shift" : "",
            stats.size(), done * n_ctx, alpha);

    gguf_free(gin);
    gguf_free(gout);
    ggml_free(meta_ctx);
    ggml_free(gctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
