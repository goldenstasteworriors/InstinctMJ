#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN_NAME=${1:-$(date +%F)_supermimic_perceptive_tracking_seed0}
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$RUN_NAME" == */* || "$RUN_NAME" == .* ]]; then
  echo "Run name must be a simple directory name" >&2
  exit 2
fi
RUN_DIR=$INSTINCT_SM_LOGS/$RUN_NAME
mkdir -p "$RUN_DIR"
exec 9>"$INSTINCT_SM_RUNTIME/tracking_train.lock"
flock -n 9 || { echo "A tracking trainer already holds the lock" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -u -m instinct_mj.supermimic.train_tracking \
  --data "$INSTINCT_SM_DATA/20251116_50cm_kneeClimbStep1" --log-dir "$RUN_DIR" "$@" \
  > >(tee "$RUN_DIR/stdout.log") 2>&1
CHECKPOINT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$RUN_DIR/last.json")
python -u -m instinct_mj.supermimic.evaluate_tracking --checkpoint "$CHECKPOINT" \
  --output "$RUN_DIR/final_eval.json" --num-envs "${INSTINCT_SM_EVAL_ENVS:-128}" \
  --steps "${INSTINCT_SM_EVAL_STEPS:-1000}" > >(tee "$RUN_DIR/eval.log") 2>&1
