#!/bin/sh
# Make a private copy of GCC 11's bits/std_function.h that nvcc 11.4 can parse (Tesla K80 / sm_37 build).
# nvcc 11.4's front end drops the parameter-pack expansion in two constrained templates and fails with
#   bits/std_function.h:435: error: parameter packs not expanded with '...'
# The copy drops those two constraints and their noexcept specs; neither matters to ggml-cuda. It is put before
# GCC's own headers with -isystem, for the CUDA compiles only. The installed header is not touched.
#   usage: scripts/k80/gcc11fix.sh [GCC11_PREFIX] [OUT_DIR]   (defaults /opt/gcc11-11.5.0, build-k80-gcc11fix)
set -e
G=${1:-/opt/gcc11-11.5.0}
O=${2:-build-k80-gcc11fix}
SRC=$(ls -d "$G"/include/c++/11*/bits/std_function.h | head -1)
[ -f "$SRC" ] || { echo "no std_function.h under $G"; exit 1; }
mkdir -p "$O/bits"
sed -e 's|typename _Constraints = _Requires<_Callable<_Functor>>>|typename _Constraints = void>  /* nvcc 11.4 */|' \
    -e 's|_Requires<_Callable<_Functor>, function&>|function\&  /* nvcc 11.4 */|' \
    -e 's|^\(\s*\)noexcept(_Handler<_Functor>::template _S_nothrow_init<_Functor>())|\1/* noexcept removed for nvcc 11.4 */|' \
    "$SRC" > "$O/bits/std_function.h"
n=$(grep -c "nvcc 11.4" "$O/bits/std_function.h")
[ "$n" -eq 4 ] || { echo "expected 4 edits, made $n: check the GCC version"; exit 1; }
echo "$O/bits/std_function.h (4 edits)"
