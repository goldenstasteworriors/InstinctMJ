#!/usr/bin/env bash
set -euo pipefail
base="$(dirname "$(realpath "$0")")"
service_bin=/nix/ykj/instinct_locomotion/service-env/bin
config="$base/supervisor_20000.conf"
case "${1:-status}" in
  start)
    if "$service_bin/supervisorctl" -c "$config" pid >/dev/null 2>&1; then
      "$service_bin/supervisorctl" -c "$config" status
      exit 0
    fi
    for name in 2026-10-29_smooth_resume8000_to20000_seed42 2026-10-29_smooth_officialflags_fresh20000_seed42; do
      run_dir="/workspace/ykj/instinct_supermimic/logs/locomotion_official/$name"
      if [[ -d "$run_dir" ]] && [[ -n "$(ls -A "$run_dir")" ]]; then
        echo "Refusing to relaunch into existing output: $run_dir" >&2
        exit 1
      fi
    done
    mkdir -p /nix/ykj/instinct_locomotion/services /nix/ykj/instinct_locomotion/build
    "$service_bin/supervisord" -c "$config"
    "$service_bin/supervisorctl" -c "$config" status
    ;;
  status) exec "$service_bin/supervisorctl" -c "$config" status ;;
  *) echo 'Usage: bash tools/supermimic_locomotion/service_20000.sh {start|status}' >&2; exit 2 ;;
esac
