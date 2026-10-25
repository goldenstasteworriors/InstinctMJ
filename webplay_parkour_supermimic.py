"""Launch the viser web viewer for the SuperMimic-trained parkour policy.

Usage: python webplay_parkour_supermimic.py [checkpoint]
(default: the newest model_*.pt of the newest run under logs/instinct_rl/g1_parkour_supermimic/). Runs the task's Play config
(``Instinct-Parkour-Target-Amp-SuperMimic-G1-Play-v0``) through ``run_play``, without tyro parsing the env cfg.
"""
import glob
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import instinct_mj.tasks  # noqa: F401  (register tasks)
from instinct_mj.scripts.instinct_rl.play import PlayConfig, run_play

TASK = "Instinct-Parkour-Target-Amp-SuperMimic-G1-Play-v0"
RUNS_GLOB = "logs/instinct_rl/g1_parkour_supermimic/*supermimic*"

if len(sys.argv) > 1:
    CKPT = sys.argv[1]
else:
    runs = sorted(glob.glob(RUNS_GLOB))
    if not runs:
        raise FileNotFoundError(f"No run matches {RUNS_GLOB}; pass a checkpoint path.")
    CKPT = max(glob.glob(f"{runs[-1]}/model_*.pt"), key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
print(f"[webplay] checkpoint: {CKPT}", flush=True)
run_play(TASK, PlayConfig(viewer="viser", checkpoint_file=CKPT))
