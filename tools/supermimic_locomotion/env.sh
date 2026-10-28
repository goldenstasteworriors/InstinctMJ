#!/usr/bin/env bash
# Source this file; dependencies are installed by `uv sync --frozen` in this UV environment.
export INSTINCT_LOCO_RUNTIME="${INSTINCT_LOCO_RUNTIME:-/nix/ykj/instinct_locomotion}"
export INSTINCT_LOCO_ROOT="${INSTINCT_LOCO_ROOT:-$INSTINCT_LOCO_RUNTIME/InstinctMJ}"
export INSTINCT_LOCO_LOGS="${INSTINCT_LOCO_LOGS:-/workspace/ykj/instinct_supermimic/logs/locomotion_official}"
export INSTINCT_PARKOUR_MOTION_DIR="${INSTINCT_PARKOUR_MOTION_DIR:-/workspace/ykj/instinct_supermimic/assets/parkour_amp}"
export INSTINCT_PARKOUR_SELECTION="${INSTINCT_PARKOUR_SELECTION:-$INSTINCT_PARKOUR_MOTION_DIR/selection.yaml}"
export INSTINCT_HEIGHTMAP_CACHE="${INSTINCT_HEIGHTMAP_CACHE:-/workspace/ykj/instinct_supermimic/assets/locomotion_heightmaps}"
export UV_PROJECT_ENVIRONMENT="$INSTINCT_LOCO_RUNTIME/.venv"
export UV_CACHE_DIR="$INSTINCT_LOCO_RUNTIME/cache/uv"
export TMPDIR="$INSTINCT_LOCO_RUNTIME/build"
export XDG_CACHE_HOME="$INSTINCT_LOCO_RUNTIME/cache/xdg"
export PYTHONPYCACHEPREFIX="$INSTINCT_LOCO_RUNTIME/cache/pycache"
export WARP_CACHE_PATH="$INSTINCT_LOCO_RUNTIME/cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$INSTINCT_LOCO_RUNTIME/cache/torch"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MUJOCO_GL=egl
export PATH="$UV_PROJECT_ENVIRONMENT/bin:/nix/ykj/supermimic/bin:$PATH"
# Existing environment-local GL runtime; no system/driver modifications.
export LD_LIBRARY_PATH="/nix/ykj/instinct_supermimic/.venv/libgl/usr/lib/x86_64-linux-gnu:/nix/ykj/supermimic/egl/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"
cd "$INSTINCT_LOCO_ROOT" || return 1
