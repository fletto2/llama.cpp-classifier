# llama-kv-fold

Calibrates a KV-cache fold and writes a copy of the model that carries it: per attention layer, key head and channel,
a scale (`blk.N.attn_k_fold`) and an offset (`blk.N.attn_k_shift`). llama.cpp then stores keys as `(k - shift) / fold`
and scales the queries by `fold`, which leaves attention unchanged and makes the keys much cheaper to quantize.
See the "KV-cache fold" section of the top-level README for the method and measurements.

```sh
llama-kv-fold -m model.gguf -f calib.txt -o model-kvfold.gguf [-c 512] [--chunks 64] [--alpha 0.5] [--no-shift] [-t N]
```

- `-c`: tokens per calibration chunk (the first token of each, the attention sink, is left out)
- `--chunks`: number of chunks read from the text
- `--alpha`: fold = rms_k^alpha / rms_q^(1-alpha) (0.5 balances key range against query weight)
- `--no-shift`: write only the fold

The input model must not already carry a fold.
