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

## Performance notes (one die, 562 MHz, 100-125 W cap)

| Qwen3-1.7B | pp512 | tg |
|---|--:|--:|
| Q8_0, CUDA | 260 t/s | 15.5 t/s (33 GB/s of weights) |
| F16, CUDA | | 10.2 t/s (41 GB/s) |
| Q8_0, Vulkan (`-DGGML_VULKAN=ON`, no patches) | 217 t/s | 16.3 t/s |

Decode reads weights at a seventh of the die's ~240 GB/s, so it is not memory-bound; the work on this branch is
finding where the time goes. Kepler has no `__dp4a` (ggml emulates it with byte extracts and integer multiplies) and
no tensor cores; FP64 runs at a third of FP32.
