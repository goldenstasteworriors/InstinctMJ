"""Finite, learning-free parkour evaluation using real simulator rewards/contacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from instinct_mj.envs import InstinctRlEnv
from instinct_mj.supermimic.config import make_config
from instinct_mj.supermimic.networks import Policy
from instinct_mj.supermimic.train import GOAL_DIM, PROPRIO_DIM, state_from_env


@torch.no_grad()
def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=1234)
  args = parser.parse_args()
  torch.set_num_threads(8)
  torch.manual_seed(args.seed)
  ck = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
  train = ck["args"]
  cfg = make_config(train["data"], args.num_envs, train["terrain_rows"], train["terrain_cols"], train["seed"])
  # Keep exactly the training terrain realization, use fresh sampling/noise.
  cfg.seed = args.seed
  env = InstinctRlEnv(cfg, device="cuda:0")
  obs, _ = env.reset()
  state = state_from_env(env, obs)
  history = state[:, None].expand(-1, 10, -1).clone()
  pi = Policy(PROPRIO_DIM, GOAL_DIM).to("cuda:0")
  pi.load_state_dict(ck["pi"])
  pi.eval()
  columns = train["terrain_cols"]
  sums = {k: torch.zeros(columns, device="cuda:0") for k in
    ("transitions", "reward_sum", "terminations", "episode_ends", "velocity_error_squared", "contact_body_sum")}
  counts_by_body = torch.zeros(env.scene["supermimic_contacts"].data.force.shape[1], device="cuda:0")
  reward_terms = {k: 0.0 for k in ck["protocol"]["reward_terms"]}
  for _ in range(args.steps):
    column = env.scene.terrain.terrain_types.clone().long()
    action = pi(history[:, :, :PROPRIO_DIM], state[:, PROPRIO_DIM:PROPRIO_DIM + GOAL_DIM])
    cmd = env.command_manager.get_command("base_velocity").clone()
    obs, reward, terminated, truncated, _ = env.step(action)
    ns = state_from_env(env, obs)
    r = reward["rewards"] if isinstance(reward, dict) else reward
    contact = env.scene["supermimic_contacts"].data.force.norm(dim=-1) > 1
    errors = (cmd[:, :2] - env.scene["robot"].data.root_link_lin_vel_b[:, :2]).square().sum(-1)
    values = (torch.ones_like(r), r, terminated.float(), (terminated | truncated).float(),
      errors, contact.float().sum(-1))
    for total, value in zip(sums.values(), values):
      total.scatter_add_(0, column, value)
    counts_by_body += contact.float().sum(0)
    for name in reward_terms:
      reward_terms[name] += float(env.reward_manager._termwise_reward_buf["rewards"][name].sum()) * env.step_dt
    history = torch.cat((history[:, 1:], ns[:, None]), 1)
    history[terminated | truncated] = ns[terminated | truncated, None]
    state = ns
  output = {"checkpoint": args.checkpoint, "iteration": ck["iteration"], "seed": args.seed,
    "steps": args.steps, "num_envs": args.num_envs,
    "protocol": "Real parkour evaluation on training terrain families; no exploration or model updates. Not held-out terrain generalization.",
    "columns": [{"column": col, **{k: float(v[col]) for k, v in sums.items()}}
                for col in range(columns)],
    "reward_terms_sum": reward_terms,
    "contact_transitions_by_sensor_body_index": counts_by_body.tolist(),
    "total_reward": float(sums["reward_sum"].sum()), "total_terminations": float(sums["terminations"].sum()),
    "mean_real_reward": float(sums["reward_sum"].sum() / sums["transitions"].sum())}
  if not torch.isfinite(torch.tensor(output["mean_real_reward"])):
    raise FloatingPointError("Evaluation reward is nonfinite")
  Path(args.output).write_text(json.dumps(output, indent=2))
  print(json.dumps({k: output[k] for k in ("iteration", "mean_real_reward", "total_terminations")}), flush=True)
  env.close()


if __name__ == "__main__":
  main()
