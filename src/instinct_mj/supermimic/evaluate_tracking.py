"""Evaluate complete reference motions; capture terminal metrics before auto-reset."""

import argparse
import json
from pathlib import Path

import torch

from instinct_mj.envs import InstinctRlEnv
from instinct_mj.supermimic.tracking_config import make_tracking_config
from instinct_mj.supermimic.tracking_model import TrackingPolicy, proprio
from instinct_mj.supermimic.tracking_task import TrackingTask


@torch.no_grad()
def main():
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--output", required=True)
  p.add_argument("--num-envs", type=int, default=128)
  p.add_argument("--steps", type=int, default=1000)
  p.add_argument("--seed", type=int, default=1234)
  args = p.parse_args()
  torch.manual_seed(args.seed)
  torch.set_num_threads(8)
  ckpt = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
  cfg = make_tracking_config(ckpt["args"]["data"], args.num_envs, args.seed, evaluation=True)
  env = InstinctRlEnv(cfg, device="cuda:0")
  obs, _ = env.reset()
  task = TrackingTask(env)
  policy = TrackingPolicy().cuda().eval()
  policy.load_state_dict(ckpt["policy"])
  state, ref = task.state(obs), task.reference()
  hist = proprio(state, task.defaults)[:, None].repeat(1, 10, 1)
  motion_ids = task.buffer._assigned_env_motion_selection.clone().long()
  names = [x["motion_file"] for x in task.buffer.yaml_data["motion_files"]]
  active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  counts = torch.zeros(env.num_envs, device=env.device)
  root_sum, link_sum, return_sum = counts.clone(), counts.clone(), counts.clone()
  contacts = torch.zeros((env.num_envs, 40), device=env.device)
  result = ["step_limit"] * env.num_envs
  terminal_terms = [[] for _ in range(env.num_envs)]
  original_compute = env.reward_manager.compute

  def capture_terminal(*a, **kw):
    value = original_compute(*a, **kw)
    d, r = task.robot.data, task.motion.reference_frame
    root_sum.add_((d.root_link_pos_w - r.base_pos_w[:, 0]).norm(dim=-1) * active)
    link_sum.add_((d.body_link_pos_w[:, task.links] - r.link_pos_w[:, 0]).norm(dim=-1).mean(-1) * active)
    contacts.add_((env.scene["contact_forces"].data.force.norm(dim=-1) > 1).float() * active[:, None])
    return value

  env.reward_manager.compute = capture_terminal
  for _ in range(args.steps):
    action = policy(hist, state, ref[:, :10], task.defaults)
    obs, reward, terminated, truncated, _ = env.step(action)
    counts.add_(active)
    r = reward["rewards"] if isinstance(reward, dict) else reward
    return_sum.add_(r * active)
    done = (terminated | truncated) & active
    for i in torch.where(done)[0].tolist():
      terms = [k for k in cfg.terminations if env.termination_manager.get_term(k)[i]]
      terminal_terms[i] = terms
      result[i] = "failure" if terminated[i] else ("motion_complete" if "dataset_exhausted" in terms else "timeout_or_invalid_reset")
    active[done] = False
    if not active.any():
      break
    state, ref = task.state(obs), task.reference()
    fresh = proprio(state, task.defaults)
    hist = torch.cat((hist[:, 1:], fresh[:, None]), 1)
    reset = terminated | truncated
    hist[reset] = fresh[reset, None].expand(-1, 10, -1)
  records = []
  for i in range(env.num_envs):
    records.append({"motion": names[int(motion_ids[i])], "result": result[i], "steps": int(counts[i]),
      "seconds": float(counts[i] * env.step_dt), "return": float(return_sum[i]),
      "root_error_m": float(root_sum[i] / counts[i].clamp_min(1)),
      "link_error_m": float(link_sum[i] / counts[i].clamp_min(1)), "termination_terms": terminal_terms[i]})
  per_motion = {}
  for name in names:
    rows = [x for x in records if x["motion"] == name]
    per_motion[name] = {"episodes": len(rows), "completed": sum(x["result"] == "motion_complete" for x in rows),
      "failed": sum(x["result"] == "failure" for x in rows),
      "mean_seconds": sum(x["seconds"] for x in rows) / max(1, len(rows)),
      "mean_root_error_m": sum(x["root_error_m"] for x in rows) / max(1, len(rows))}
  output = {"checkpoint": args.checkpoint, "iteration": ckpt["iteration"], "seed": args.seed,
    "protocol": "First episode per environment, from full-motion start, paired training meshes, randomized physics; terminal metrics captured before reset",
    "completed": result.count("motion_complete"), "failed": result.count("failure"),
    "censored": sum(x not in ("failure", "motion_complete") for x in result),
    "per_motion": per_motion, "records": records,
    "body_contact_steps": dict(zip(task.robot.body_names, contacts.sum(0).tolist())),
    "total_env_steps": int(counts.sum())}
  Path(args.output).write_text(json.dumps(output, indent=2))
  print(json.dumps({k: v for k, v in output.items() if k not in ("records", "body_contact_steps")}), flush=True)
  env.close()


if __name__ == "__main__":
  main()
