#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(realpath "$0")")/env.sh"
export CUDA_VISIBLE_DEVICES="${TRACKING_GPU:-0}"
mode="${1:-train}"
shift || true
case "$mode" in
  train)
    name="${1:-$(date +%F)_official_tracking_seed42}"
    if (($#)); then shift; fi
    [[ "$name" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[A-Za-z0-9_-]+$ ]] || exit 2
    exec 9>"$TRACKING_RUNTIME/train_gpu${CUDA_VISIBLE_DEVICES}.lock"
    flock -n 9 || { echo 'Tracking training already holds this GPU lock.' >&2; exit 1; }
    exec python -u tools/official_tracking/train.py --run-name "$name" "$@"
    ;;
  check)
    exec python -u tests/supermimic/check_analytic_heads.py --motion-folder "$TRACKING_DATA" --seed 42 "$@"
    ;;
  eval)
    exec python -u -m instinct_mj.scripts.supermimic.evaluate --motion-folder "$TRACKING_DATA" --num-envs 128 "$@"
    ;;
  *) echo 'Usage: run.sh {train [dated_name] [options]|check [options]|eval [options]}' >&2; exit 2 ;;
esac
