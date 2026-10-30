#!/usr/bin/env bash
set -euo pipefail
export INSTINCT_LOCO_ROOT=/nix/ykj/instinct_locomotion/InstinctMJ-improved
source "$INSTINCT_LOCO_ROOT/tools/supermimic_locomotion/env.sh"
export PYTHONPATH="$INSTINCT_LOCO_ROOT/src"
exec python -u tools/supermimic_locomotion/depth_repair_queue.py
