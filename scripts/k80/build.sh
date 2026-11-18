#!/bin/sh
# Build for the Tesla K80 (GK210, sm_37) with CUDA 11.4 and GCC 11 (CUDA 11.4 is the last toolkit with sm_37;
# driver 470 is the last driver for Kepler). See docs/build-k80.md.
#   usage: scripts/k80/build.sh [targets...]   (default: llama-server llama-perplexity llama-bench)
# env: CUDA (default /opt/cuda-11.4), GCC11 (default /opt/gcc11-11.5.0), JOBS (default 8), BUILD (default build-k80),
#      HF_UI_VERSION (web UI build, default: the upstream build number of the newest sync merge)
set -e
CUDA=${CUDA:-/opt/cuda-11.4}
GCC11=${GCC11:-/opt/gcc11-11.5.0}
BUILD=${BUILD:-build-k80}
FIX=$(pwd)/$BUILD-gcc11fix
# llama-server's web UI is fetched by build number (HF bucket ggml-org/llama-ui). This fork's commit count is ahead of
# upstream, so its number does not exist there and the build falls back to "latest", a newer UI this tree's asset
# check rejects (no workbox*.js). Ask for the UI of the upstream commit of the newest upstream sync merge instead.
# tools/ui/CMakeLists.txt reads it from the environment at configure time.
if [ -z "$HF_UI_VERSION" ]; then
    U=$(git log --merges -1 --format=%H --grep="origin/master. into wllm-main" 2>/dev/null)
    [ -n "$U" ] && HF_UI_VERSION=b$(git rev-list --count "$U^2") && export HF_UI_VERSION
fi
sh scripts/k80/gcc11fix.sh "$GCC11" "$FIX"
export PATH=$CUDA/bin:$PATH CUDAHOSTCXX=$GCC11/bin/g++
cmake -B "$BUILD" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=37 -DGGML_CUDA_NO_VMM=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc \
  "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -isystem $FIX -Wno-deprecated-gpu-targets" \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,$CUDA/lib64" \
  "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,$CUDA/lib64" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
[ $# -gt 0 ] || set -- llama-server llama-perplexity llama-bench
cmake --build "$BUILD" -j "${JOBS:-8}" --target "$@"
