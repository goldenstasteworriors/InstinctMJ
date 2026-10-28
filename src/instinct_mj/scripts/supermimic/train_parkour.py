"""Train the parkour task with SuperMimic (``Instinct-Parkour-Target-Amp-SuperMimic-G1-v0``).

Same training path as ``instinct-train`` (``launch_training``), but the env cfg is not parsed by tyro: the parkour terrain cfg
has defaults outside their declared types (lists of flat-patch radii where mjlab declares ``float``), which tyro rejects.
Only the run options below and the agent fields can be overridden:

    python -m instinct_mj.scripts.supermimic.train_parkour [--task ID] [--num-envs N] [--device cuda:0] \
        [--agent.<field> <value> ...]   # e.g. --agent.max-iterations 8000 --agent.load-run <run> --agent.load-checkpoint model_<it>.pt
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import instinct_mj.tasks  # noqa: F401  (register the tasks)
from instinct_mj.scripts.instinct_rl.train import TrainConfig, launch_training, run_train

DEFAULT_TASK = "Instinct-Parkour-Target-Amp-SuperMimic-G1-v0"


def _cast(text: str, default):
    """Parse ``text`` as the type of the field's default (bool, int, float, str; None defaults keep strings)."""
    if isinstance(default, bool):
        if text.lower() not in ("true", "false"):
            raise ValueError(f"expected True / False, got {text!r}")
        return text.lower() == "true"
    if isinstance(default, int):
        return int(text)
    if isinstance(default, float):
        return float(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", type=Path, default=None, help="Explicit output directory; must be empty for a new run.")
    args, rest = parser.parse_known_args()
    cfg = TrainConfig.from_task(args.task)
    agent = cfg.agent
    if len(rest) % 2 != 0:
        raise ValueError(f"agent overrides must be '--agent.<field> <value>' pairs, got {rest}")
    for flag, text in zip(rest[::2], rest[1::2]):
        if not flag.startswith("--agent."):
            raise ValueError(f"unknown argument {flag}")
        name = flag[len("--agent.") :].replace("-", "_")
        if not hasattr(agent, name):
            raise ValueError(f"the agent cfg has no field {name}")
        setattr(agent, name, _cast(text, getattr(agent, name)))
    agent.resume = agent.load_run is not None  # set by the cfg's __post_init__ from the default load_run
    cfg = dataclasses.replace(cfg, agent=agent, num_envs=args.num_envs, device=args.device)
    if args.log_dir is None:
        launch_training(args.task, cfg)
    else:
        if not agent.resume and args.log_dir.exists() and any(args.log_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty run directory: {args.log_dir}")
        run_train(args.task, cfg, args.log_dir)


if __name__ == "__main__":
    main()
