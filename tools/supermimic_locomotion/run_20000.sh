#!/usr/bin/env bash
set -euo pipefail
base="$(dirname "$(realpath "$0")")"
case "${1:-}" in
  resume)
    exec env INSTINCT_LOCO_GPU=0 bash "$base/run.sh" train \
      2026-10-29_smooth_resume8000_to20000_seed42 \
      --agent.max-iterations 20000 \
      --agent.load-run /workspace/ykj/instinct_supermimic/logs/locomotion_official/2026-10-28_supermimic_parkour_seed42 \
      --agent.load-checkpoint model_8000.pt \
      --agent.fail-head True --agent.wm-failure-frac 0.25 --agent.wm-holdout-envs 0.1
    ;;
  fresh)
    exec env INSTINCT_LOCO_GPU=2 bash "$base/run.sh" train \
      2026-10-29_smooth_officialflags_fresh20000_seed42 \
      --agent.max-iterations 20000 \
      --agent.fail-head False --agent.wm-failure-frac 0 --agent.wm-holdout-envs 0
    ;;
  *) echo 'Usage: bash tools/supermimic_locomotion/run_20000.sh {resume|fresh}' >&2; exit 2 ;;
esac
