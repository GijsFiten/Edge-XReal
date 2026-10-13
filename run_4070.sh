#!/usr/bin/env bash
set -euo pipefail
: "${CONDA_PREFIX:?Activate conda hsu first}"
if [[ "$(basename "$CONDA_PREFIX")" != hsu ]]; then
  echo 'Activate the hsu environment first.' >&2; exit 1
fi
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export TORCH_CUDA_ARCH_LIST=8.9
if [[ -e /usr/lib/wsl/lib/libcuda.so.1 ]]; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PATH="$CONDA_PREFIX/bin:$PATH"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "$CONDA_PREFIX/bin/python" "$@"
