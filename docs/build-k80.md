# Tesla K80 (Kepler, sm_37) with CUDA 11.4

Branch `k80-cuda11`. CUDA 11.4 is the last toolkit that targets sm_37 and driver 470 the last driver for Kepler.

## Build

    scripts/k80/build.sh                      # llama-server, llama-perplexity, llama-bench into build-k80/

Needs CUDA 11.4 (`CUDA=`) and GCC 11 as the host compiler (`GCC11=`). CUDA 11.4 officially supports GCC up to 10;
GCC 11 works with `-allow-unsupported-compiler` and a private copy of `bits/std_function.h` that
`scripts/k80/gcc11fix.sh` makes from the installed GCC (nvcc 11.4 cannot parse two of its constrained templates). The
copy is used for the CUDA compiles only. Other flags:
- `GGML_CUDA_NO_VMM=ON`: Kepler has no virtual memory management.
- The rpath to `$CUDA/lib64`, which is usually not in ld.so.conf.
- A full build takes about an hour on a loaded 32-thread machine; most of it is the template-instance files.

## Run

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 build-k80/bin/llama-server -m MODEL.gguf -ngl 99

Each K80 board is two GPUs (dies) of 12 GB. `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA's numbering match nvidia-smi.

## Performance notes

Each die drops to P8 (324 MHz core and memory, about 31 GB/s) whenever its power cap binds, even at 97% utilization;
nvidia-smi does not report the cap as active when it samples. At 100 W (the minimum) a die spends most of a decode in
P8; at 125 W it holds P0 (562 MHz core, 2,505 MHz memory) under load. Check `nvidia-smi --query-gpu=pstate,clocks.mem`
during a run before trusting a speed number. `nvidia-smi --cuda-clocks` is Volta+ only.

One die at 100 W (mostly P8):

| Qwen3-1.7B | pp512 | tg |
|---|--:|--:|
| Q8_0, CUDA | 260 t/s | 15.5 t/s (33 GB/s of weights) |
| F16, CUDA | | 10.2 t/s (41 GB/s) |
| Q8_0, Vulkan (`-DGGML_VULKAN=ON`, no patches) | 217 t/s | 16.3 t/s |

Kepler has no `__dp4a` (ggml emulates it with byte extracts and integer multiplies), no tensor cores, and FP64 at a
third of FP32. Kernel work on this branch is measured on a die that holds P0.

## Kepler Q8_0 decode kernel (`ggml/src/ggml-cuda/mmv-kepler.cu`)

For compute capability < 6.1, a Q8_0 x F32 matrix-vector product (batch 1, 2D, ne00 % 64 == 0) replaces MMVQ, whose
dp4a emulation reaches ~40 GB/s on a K80 die. It converts the int8 weights to float and FMAs them with the F32
activations, reading each 34-byte block as 9 aligned words (`__byte_perm` for odd blocks), 4 rows per warp (8 when
ne00 >= 8192) so each x load serves several rows. `GGML_CUDA_NO_KEPLER_MMV=1` turns it off.

One die at 125 W, P0 throughout:

| | stock (MMVQ) | Kepler kernel |
|---|--:|--:|
| matvec 4096 x 14336, standalone (`k80_q8mv` E8) | 40 GB/s | 150 GB/s |
| Qwen3-1.7B Q8_0 tg128 | 28.4 t/s | 50.8 t/s (1.79x) |

Accuracy (llama-perplexity, Qwen3-1.7B Q8_0, 2 x 512 tokens, micro-batch 1 so every token goes through the kernel,
KL against stock MMVQ with the same settings): mean KLD 0.00109, same top token 97.6%, PPL ratio 0.9985 +/- 0.0025.
Stock batched vs stock micro-batch 1 differ by the same amount (0.00109, 97.5%). test-backend-ops: the batch-1 Q8_0
cases pass (19/19). A full `test -o MUL_MAT` run aborts on a batched case with CUBLAS_STATUS_ARCH_MISMATCH, with or
without the kernel (open).
