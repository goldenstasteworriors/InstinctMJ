#!/usr/bin/env bash
export TRACKING_RUNTIME="${TRACKING_RUNTIME:-/nix/ykj/instinct_tracking_official}"
export TRACKING_ROOT="$TRACKING_RUNTIME/InstinctMJ"
export TRACKING_LOGS=/workspace/ykj/instinct_supermimic/logs/tracking_official
export TRACKING_DATA=/workspace/ykj/instinct_supermimic/assets/tracking_official_release/20251116_50cm_kneeClimbStep1
export INSTINCT_HEIGHTMAP_CACHE=/workspace/ykj/instinct_supermimic/assets/tracking_official_heightmaps
export UV_PROJECT_ENVIRONMENT="$TRACKING_RUNTIME/.venv"
# Reuse downloaded, hash-checked package cache, not another project's environment.
export UV_CACHE_DIR=/nix/ykj/instinct_locomotion/cache/uv
export TMPDIR="$TRACKING_RUNTIME/build"
export XDG_CACHE_HOME="$TRACKING_RUNTIME/cache/xdg"
export PYTHONPYCACHEPREFIX="$TRACKING_RUNTIME/cache/pycache"
export WARP_CACHE_PATH="$TRACKING_RUNTIME/cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$TRACKING_RUNTIME/cache/torch"
export TRITON_CACHE_DIR="$TRACKING_RUNTIME/cache/triton"
export CUDA_CACHE_PATH="$TRACKING_RUNTIME/cache/cuda"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl
export PATH="$UV_PROJECT_ENVIRONMENT/bin:/nix/ykj/supermimic/bin:$PATH"
export LD_LIBRARY_PATH="/nix/ykj/instinct_supermimic/.venv/libgl/usr/lib/x86_64-linux-gnu:/nix/ykj/supermimic/egl/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TRACKING_ROOT" || return 1
