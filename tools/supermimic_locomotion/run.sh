#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$(realpath "$0")")/env.sh"
mode="${1:-train}"
shift || true
export CUDA_VISIBLE_DEVICES="${INSTINCT_LOCO_GPU:-1}"
case "$mode" in
  check)
    exec python -u tests/supermimic/check_parkour_heads.py "$@"
    ;;
  eval)
    exec python -u -m instinct_mj.scripts.supermimic.evaluate_parkour "$@"
    ;;
  train)
    name="${1:-$(date +%F)_supermimic_parkour_seed42}"
    if (($#)); then shift; fi
    [[ "$name" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[A-Za-z0-9_-]+$ ]] || { echo 'Run name must start with YYYY-MM-DD_ and contain only letters, digits, _ or -.' >&2; exit 2; }
    mkdir -p "$INSTINCT_LOCO_LOGS"
    exec 9>"$INSTINCT_LOCO_RUNTIME/train_gpu${CUDA_VISIBLE_DEVICES}.lock"
    flock -n 9 || { echo 'A locomotion training job already holds this GPU lock.' >&2; exit 1; }
    exec python -u -m instinct_mj.scripts.supermimic.train_parkour \
      --num-envs 1024 --device cuda:0 --log-dir "$INSTINCT_LOCO_LOGS/$name" \
      --agent.seed 42 --agent.run-name "$name" --agent.max-iterations 8000 \
      --agent.save-interval 500 --agent.heightmap-cache-dir "$INSTINCT_HEIGHTMAP_CACHE" "$@"
    ;;
  *) echo 'Usage: bash tools/supermimic_locomotion/run.sh {check [options]|eval --checkpoint FILE --output FILE [options]|train [YYYY-MM-DD_name] [options]}' >&2; exit 2 ;;
esac
