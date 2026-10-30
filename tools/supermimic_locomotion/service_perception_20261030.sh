#!/usr/bin/env bash
set -euo pipefail
base="$(dirname "$(realpath "$0")")"
export INSTINCT_LOCO_ROOT="$(realpath "$base/../..")"
source "$base/env.sh"
export PYTHONPATH="$INSTINCT_LOCO_ROOT/src"
config="$base/supervisor_perception_20261030.conf"
service_bin="$INSTINCT_LOCO_RUNTIME/service-env/bin"
case "${1:-status}" in
  worker) exec python -u "$base/perception_queue.py" run ;;
  prepare) exec python "$base/perception_queue.py" prepare ;;
  start)
    python "$base/perception_queue.py" prepare
    if "$service_bin/supervisorctl" -c "$config" pid >/dev/null 2>&1; then
      exec "$service_bin/supervisorctl" -c "$config" status
    fi
    mkdir -p "$INSTINCT_LOCO_RUNTIME/services"
    "$service_bin/supervisord" -c "$config"
    "$service_bin/supervisorctl" -c "$config" status
    ;;
  status)
    python "$base/perception_queue.py" status
    exec "$service_bin/supervisorctl" -c "$config" status
    ;;
  *) echo 'Usage: service_perception_20261030.sh {prepare|start|status|worker}' >&2; exit 2 ;;
esac
