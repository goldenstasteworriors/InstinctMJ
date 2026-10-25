#!/usr/bin/env bash
export INSTINCT_SM_RUNTIME=/nix/ykj/instinct_supermimic
export INSTINCT_SM_ROOT=$INSTINCT_SM_RUNTIME/InstinctMJ
export INSTINCT_SM_DATA=/workspace/ykj/instinct_supermimic/assets
export INSTINCT_SM_LOGS=/workspace/ykj/instinct_supermimic/logs
export TMPDIR=$INSTINCT_SM_RUNTIME/build
export UV_CACHE_DIR=$INSTINCT_SM_RUNTIME/cache/uv
export XDG_CACHE_HOME=$INSTINCT_SM_RUNTIME/cache/xdg
export PYTHONPYCACHEPREFIX=$INSTINCT_SM_RUNTIME/cache/pycache
export WARP_CACHE_PATH=$INSTINCT_SM_RUNTIME/cache/warp
export TORCHINDUCTOR_CACHE_DIR=$INSTINCT_SM_RUNTIME/cache/torch
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH=$INSTINCT_SM_RUNTIME/.venv/bin:/nix/ykj/supermimic/bin:$PATH
export LD_LIBRARY_PATH=$INSTINCT_SM_RUNTIME/.venv/libgl/usr/lib/x86_64-linux-gnu:/nix/ykj/supermimic/egl/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
cd "$INSTINCT_SM_ROOT"
