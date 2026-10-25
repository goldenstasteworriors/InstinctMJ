#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN_NAME=${1:-$(date +%F)_supermimic_parkour_seed0}
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$RUN_NAME" == */* || "$RUN_NAME" == .* ]]; then
  echo "Run name must be a simple directory name" >&2
  exit 2
fi
RUN_DIR=$INSTINCT_SM_LOGS/$RUN_NAME
mkdir -p "$RUN_DIR"
exec 9>"$INSTINCT_SM_RUNTIME/train.lock"
flock -n 9 || { echo "An Instinct SuperMimic trainer already holds the lock" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -u -m instinct_mj.supermimic.train \
  --data "$INSTINCT_SM_DATA/parkour_amp" --log-dir "$RUN_DIR" "$@" \
  > >(tee "$RUN_DIR/stdout.log") 2>&1
