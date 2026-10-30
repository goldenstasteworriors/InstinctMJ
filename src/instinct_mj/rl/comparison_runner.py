"""Unmodified PPO updates with scoped checkpoint retention and sample accounting."""

import json
import math
from pathlib import Path
import statistics

from instinct_rl.runners import OnPolicyRunner


class ComparisonPPORunner(OnPolicyRunner):
    def log(self, locs, *args, **kwargs):
        super().log(locs, *args, **kwargs)
        if not self.log_dir:
            return
        rewards = locs["rewbuffer"]
        self.audit_score = statistics.mean(rewards[0]) if rewards and rewards[0] else None
        status = dict(iteration=self.current_learning_iteration,
                      real_transitions=self.tot_timesteps, wall_seconds=self.tot_time,
                      mean_episode_reward=self.audit_score,
                      mean_episode_steps=statistics.mean(locs["lenbuffer"]) if locs["lenbuffer"] else None)
        Path(self.log_dir, "status.json").write_text(json.dumps(status))

    def save(self, path, infos=None):
        super().save(path, infos)
        directory = Path(path).parent.resolve()
        manifest_path = directory / "checkpoints.json"
        saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(managed=[], best=None, score=None)
        name = Path(path).name
        if name not in saved["managed"]:
            saved["managed"].append(name)
        score = getattr(self, "audit_score", None)
        if score is not None and math.isfinite(score) and (saved["score"] is None or score > saved["score"]):
            saved.update(best=name, score=score)
        keep = set(saved["managed"][-2:]) | {saved["best"]}
        for old in list(saved["managed"]):
            if old in keep:
                continue
            target = directory / old
            if target.is_symlink() or target.resolve().parent != directory or not old.startswith("model_"):
                raise ValueError("Unsafe managed checkpoint")
            target.unlink(missing_ok=True)
            saved["managed"].remove(old)
        saved.update(latest=name, iteration=self.current_learning_iteration,
                     best_metric="training episode reward (proxy, not independent validation)")
        manifest_path.write_text(json.dumps(saved, indent=2))
