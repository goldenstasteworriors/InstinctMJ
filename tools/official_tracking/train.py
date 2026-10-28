"""Official published-result settings; only deployment paths and output name are adapted."""

import argparse
import dataclasses
import os
from pathlib import Path

from instinct_mj.scripts.instinct_rl.train import TrainConfig, run_train
from instinct_mj.scripts.supermimic.common import TASK_ID, apply_motion_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--validation", action="store_true", help="Four full-size updates with two model-only updates.")
    args = parser.parse_args()
    cfg = dataclasses.replace(TrainConfig.from_task(TASK_ID), num_envs=1024, device="cuda:0")
    apply_motion_folder(cfg.env, os.environ["TRACKING_DATA"])
    cfg.agent.max_iterations = 8000
    cfg.agent.fail_head = False
    cfg.agent.wm_failure_frac = 0.0
    cfg.agent.wm_holdout_envs = 0.0
    cfg.agent.run_name = args.run_name
    cfg.agent.heightmap_cache_dir = os.environ["INSTINCT_HEIGHTMAP_CACHE"]
    if args.validation:
        cfg.agent.max_iterations, cfg.agent.wm_only_iters = 4, 2
        cfg.agent.log_interval, cfg.agent.save_interval = 1, 1
    log_dir = Path(os.environ["TRACKING_LOGS"], args.run_name)
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {log_dir}")
    run_train(TASK_ID, cfg, log_dir)


if __name__ == "__main__":
    main()
