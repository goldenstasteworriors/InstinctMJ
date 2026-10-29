#!/usr/bin/env bash
set -euo pipefail
base="$(dirname "$(realpath "$0")")"
source "$base/env.sh"
config="$base/supervisor_queue_20261030.conf"
service_bin="$INSTINCT_LOCO_RUNTIME/service-env/bin"
case "${1:-status}" in
  prepare) exec python "$base/experiment_queue.py" prepare ;;
  worker) exec python -u "$base/experiment_queue.py" run ;;
  start)
    python "$base/experiment_queue.py" prepare
    if "$service_bin/supervisorctl" -c "$config" pid >/dev/null 2>&1; then
      "$service_bin/supervisorctl" -c "$config" status
      exit 0
    fi
    mkdir -p "$INSTINCT_LOCO_RUNTIME/services"
    "$service_bin/supervisord" -c "$config"
    "$service_bin/supervisorctl" -c "$config" status
    ;;
  status)
    python "$base/experiment_queue.py" status
    exec "$service_bin/supervisorctl" -c "$config" status
    ;;
  *) echo 'Usage: bash tools/supermimic_locomotion/service_queue_20261030.sh {prepare|start|status|worker}' >&2; exit 2 ;;
esac
