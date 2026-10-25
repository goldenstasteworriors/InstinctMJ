"""Collect native parkour transitions, fit dynamics, optimize policy through imagination.

This is a SuperMimic-style parkour extension, not the flat-ground reproduction.
Original sensor-dependent task terms are supervised from mjlab's reward manager;
simple kinematic terms remain analytic during imagination. AMP is kept as a
separately trained least-squares discriminator with the original quad reward.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from instinct_mj.envs import InstinctRlEnv
from instinct_mj.supermimic.config import make_config
from instinct_mj.supermimic.networks import Critic, Policy, RunningNorm, WorldModel, mlp
from instinct_mj.supermimic.replay import ReplayBuffer

PROPRIO_DIM = 93
DEPTH_DIM = 9 * 16
GOAL_DIM = 3 + DEPTH_DIM
AMP_DIM = 67


def flatten_amp(group: dict[str, torch.Tensor]) -> torch.Tensor:
  return torch.cat([group[k].flatten(1) for k in
    ("projected_gravity", "joint_pos_rel", "joint_vel", "base_lin_vel", "base_ang_vel")], -1)


def state_from_env(env: InstinctRlEnv, obs: dict) -> torch.Tensor:
  group = obs["policy"]
  proprio = torch.cat([group[k].reshape(env.num_envs, -1)[:, -dim:] for k, dim in
    (("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", 29), ("joint_vel", 29), ("actions", 29))], -1)
  depth = group["depth_image"]
  # Native depth component is [N, history, H, W] (or includes a singleton channel).
  if depth.ndim < 3:
    raise ValueError(f"Unexpected depth shape: {depth.shape}")
  depth = depth.reshape(env.num_envs, -1, depth.shape[-2], depth.shape[-1])[:, -1:]
  depth = F.adaptive_avg_pool2d(depth, (9, 16)).flatten(1)
  robot = env.scene["robot"].data
  contact = env.scene["supermimic_contacts"].data.force
  force = torch.log1p(contact.norm(dim=-1).clamp_max(1e5))
  goal = torch.cat((env.command_manager.get_command("base_velocity"), depth), -1)
  privileged = torch.cat((robot.root_link_lin_vel_b, robot.root_link_pos_w - env.scene.env_origins,
    force.flatten(1), robot.applied_torque), -1)
  state = torch.cat((proprio, goal, privileged), -1).float()
  if not torch.isfinite(state).all():
    raise FloatingPointError("Nonfinite simulator state/depth/contact")
  return state


def amp_from_history(history: torch.Tensor) -> torch.Tensor:
  # The discriminator component order is term-major, exactly as flatten_amp().
  return torch.cat((history[..., 3:6].flatten(1), history[..., 6:35].flatten(1),
    history[..., 35:64].flatten(1), history[..., PROPRIO_DIM + GOAL_DIM:PROPRIO_DIM + GOAL_DIM + 3].flatten(1),
    (history[..., :3] / 0.25).flatten(1)), -1)


def imagined_reward(state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor,
                    terms: torch.Tensor, names: list[str], dt: float) -> torch.Tensor:
  rewards = []
  cmd = state[:, PROPRIO_DIM:PROPRIO_DIM + 3]
  v = next_state[:, PROPRIO_DIM + GOAL_DIM:PROPRIO_DIM + GOAL_DIM + 3]
  w = next_state[:, :3] / 0.25
  for i, name in enumerate(names):
    if name == "track_lin_vel_xy_exp":
      value = 2 * torch.exp(-(cmd[:, :2] - v[:, :2]).square().sum(-1) / 0.25) * dt
    elif name == "track_ang_vel_z_exp":
      value = 2 * torch.exp(-(cmd[:, 2] - w[:, 2]).square() / 0.25) * dt
    elif name == "heading_error":
      value = -cmd[:, 2].abs() * dt
    elif name == "is_alive":
      value = torch.full_like(cmd[:, 0], 3 * dt)
    elif name == "action_rate_l2":
      value = -0.005 * (action - state[:, 64:93]).square().sum(-1) * dt
    elif name == "dof_vel_l2":
      value = -1e-4 * (next_state[:, 35:64] / 0.05).square().sum(-1) * dt
    elif name == "flat_orientation_l2":
      value = -3 * next_state[:, 3:5].square().sum(-1) * dt
    elif name == "feet_air_time":
      value = terms[:, i].clamp(0, 10 * dt)
    elif name == "feet_close_xy":
      value = terms[:, i].clamp(0, 0.4 * dt)
    elif name == "stand_still":
      value = terms[:, i].clamp(max=1.2 * dt)
    else:
      value = terms[:, i].clamp(max=0)
    rewards.append(value)
  return torch.stack(rewards, -1).sum(-1)


def optimize(loss: torch.Tensor, optimizer: torch.optim.Optimizer, model: nn.Module) -> float:
  if not torch.isfinite(loss):
    raise FloatingPointError("Nonfinite loss; optimizer was not advanced")
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
  optimizer.step()
  return float(norm)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--data", required=True)
  parser.add_argument("--log-dir", required=True)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--terrain-rows", type=int, default=5)
  parser.add_argument("--terrain-cols", type=int, default=20)
  parser.add_argument("--iterations", type=int, default=320000)
  parser.add_argument("--max-hours", type=float, default=12)
  parser.add_argument("--buffer-steps", type=int, default=512)
  parser.add_argument("--warmup-steps", type=int, default=64)
  parser.add_argument("--wm-only-iters", type=int, default=500)
  parser.add_argument("--wm-window", type=int, default=8)
  parser.add_argument("--wm-batches", type=int, default=10)
  parser.add_argument("--wm-batch", type=int, default=512)
  parser.add_argument("--pi-batch", type=int, default=512)
  parser.add_argument("--width", type=int, default=1024)
  parser.add_argument("--horizon", type=int, default=32)
  parser.add_argument("--hist", type=int, default=10)
  parser.add_argument("--collect-steps", type=int, default=3)
  parser.add_argument("--lr", type=float, default=2e-4)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--lam", type=float, default=0.95)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--ckpt-every", type=int, default=1000)
  parser.add_argument("--log-every", type=int, default=25)
  parser.add_argument("--resume")
  args = parser.parse_args()
  if args.hist != 10:
    raise ValueError("Native AMP histories require --hist 10")
  torch.manual_seed(args.seed)
  torch.set_num_threads(8)
  torch.backends.cuda.matmul.allow_tf32 = True
  device = "cuda:0"
  logdir = Path(args.log_dir)
  logdir.mkdir(parents=True, exist_ok=True)
  (logdir / "args.json").write_text(json.dumps(vars(args), indent=2))
  cfg = make_config(args.data, args.num_envs, args.terrain_rows, args.terrain_cols, args.seed)
  env = InstinctRlEnv(cfg, device=device)
  obs, _ = env.reset()
  state = state_from_env(env, obs)
  state_dim = state.shape[-1]
  term_names = list(cfg.rewards["rewards"] if "rewards" in cfg.rewards else cfg.rewards)
  # MultiRewardManager groups the original flat native task terms.
  term_names = list(env.reward_manager._termwise_reward_buf["rewards"])
  wm = WorldModel(state_dim, 29, len(term_names), args.width).to(device)
  pi = Policy(PROPRIO_DIM, GOAL_DIM, args.hist).to(device)
  critic = Critic(state_dim).to(device)
  disc = mlp(AMP_DIM * args.hist, [1024, 512], 1).to(device)
  disc_norm = RunningNorm(AMP_DIM * args.hist).to(device)
  models = {"wm": wm, "pi": pi, "critic": critic, "disc": disc, "disc_norm": disc_norm}
  optimizers = {name: torch.optim.Adam(model.parameters(), lr=args.lr,
      betas=(0.9, 0.95) if name == "pi" else (0.9, 0.999))
      for name, model in models.items() if name != "disc_norm"}
  optimizers["disc"] = torch.optim.AdamW(disc.parameters(), lr=1e-4, weight_decay=3e-4)
  start, wall_offset, sim_steps = 0, 0.0, 0
  if args.resume:
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    for name, model in models.items():
      model.load_state_dict(checkpoint[name])
    for name, optimizer in optimizers.items():
      optimizer.load_state_dict(checkpoint["optimizers"][name])
    start, wall_offset, sim_steps = checkpoint["iteration"], checkpoint["wall_time"], checkpoint["sim_steps"]
    for model in models.values():
      if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise FloatingPointError("Refusing poisoned resume checkpoint")
  protocol = {"variant": "supermimic_parkour_sensor_reward_model_v1", "task": "Instinct-Parkour-Target-Amp-G1-v0",
    "state_dim": state_dim, "proprio_dim": PROPRIO_DIM, "goal_dim": GOAL_DIM,
    "reward_terms": term_names, "dt": env.step_dt, "original_task_reward_and_termination": True,
    "differences": ["sensor-dependent imagined reward terms are learned from original task labels",
      "world model predicts native feature increments, including pooled depth and contacts",
      "velocity command and depth replace reference-motion goals; AMP style prior retained",
      "mixed contact/LAFAN AMP library; not original parkour reference distribution",
      "pooled 9x16 latest depth plus 10 proprio frames; not original PPO CNN/MoE policy"],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
  (logdir / "protocol.json").write_text(json.dumps(protocol, indent=2))
  replay = ReplayBuffer(args.buffer_steps, args.num_envs, device)
  history = state[:, None].expand(-1, args.hist, -1).clone()
  ep_ret = torch.zeros(args.num_envs, device=device)
  ep_len = torch.zeros_like(ep_ret)
  completed_returns, completed_lengths = deque(maxlen=1000), deque(maxlen=1000)
  stop = False

  def request_stop(_sig, _frame):
    nonlocal stop
    stop = True

  signal.signal(signal.SIGTERM, request_stop)
  signal.signal(signal.SIGINT, request_stop)
  started = time.monotonic()
  reward_bound = (2 + 2 + 3 + 1.2 + 0.4 + 0.5 * cfg.episode_length_s) * env.step_dt + 0.25

  def save(iteration: int) -> None:
    checkpoint = {name: model.state_dict() for name, model in models.items()}
    checkpoint.update(optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
      iteration=iteration, wall_time=wall_offset + time.monotonic() - started,
      sim_steps=sim_steps, args=vars(args), protocol=protocol, torch_rng=torch.get_rng_state(),
      cuda_rng=torch.cuda.get_rng_state())
    path = logdir / f"ckpt_{iteration:06d}.pt"
    torch.save(checkpoint, path)
    # bosfs does not support replacing an open checkpoint. Write and close first.
    latest = logdir / "last.json"
    latest.write_text(json.dumps({"checkpoint": str(path), "iteration": iteration,
      "wall_time": checkpoint["wall_time"], "sim_steps": sim_steps}))
    print(f"CHECKPOINT {path}", flush=True)

  @torch.no_grad()
  def collect(steps: int) -> tuple[float, float]:
    nonlocal obs, state, history, sim_steps
    reward_mean, termination_mean = 0.0, 0.0
    for _ in range(steps):
      pi.obs_norm.update(state[:, :PROPRIO_DIM])
      pi.goal_norm.update(state[:, PROPRIO_DIM:PROPRIO_DIM + GOAL_DIM])
      critic.norm.update(state)
      action = (pi(history[:, :, :PROPRIO_DIM], state[:, PROPRIO_DIM:PROPRIO_DIM + GOAL_DIM])
                + 0.1 * torch.randn(args.num_envs, 29, device=device)).clamp(-20, 20)
      amp_actor, amp_ref = flatten_amp(obs["amp_policy"]), flatten_amp(obs["amp_reference"])
      next_obs, reward, terminated, truncated, extras = env.step(action)
      next_state = state_from_env(env, next_obs)
      done = terminated | truncated
      term_values = torch.stack([env.reward_manager._termwise_reward_buf["rewards"][name]
        for name in term_names], -1) * env.step_dt
      r = reward["rewards"] if isinstance(reward, dict) else reward
      if not torch.allclose(term_values.sum(-1), r, atol=1e-5, rtol=1e-4):
        raise AssertionError("Reward labels do not reproduce native task reward")
      replay.add(state=state, action=action, next_state=next_state, terms=term_values,
        done=done, terminated=terminated, amp_actor=amp_actor, amp_ref=amp_ref)
      wm.input_norm.update(torch.cat((state, action), -1))
      valid = ~done
      if valid.any():
        wm.delta_norm.update(next_state[valid] - state[valid])
      wm.reward_norm.update(term_values)
      disc_norm.update(torch.cat((amp_actor, amp_ref)))
      ep_ret += r
      ep_len += 1
      if done.any():
        completed_returns.extend(ep_ret[done].tolist())
        completed_lengths.extend((ep_len[done] * env.step_dt).tolist())
        ep_ret[done], ep_len[done] = 0, 0
      history = torch.cat((history[:, 1:], next_state[:, None]), 1)
      history[done] = next_state[done, None]
      obs, state = next_obs, next_state
      sim_steps += args.num_envs
      reward_mean += float(r.mean()) / steps
      termination_mean += float(terminated.float().mean()) / steps
    return reward_mean, termination_mean

  def wm_update() -> tuple[float, float]:
    batch = replay.sample(args.wm_batch, args.wm_window)
    predicted = batch["state"][:, 0]
    loss = torch.zeros((), device=device)
    last_error = torch.zeros((), device=device)
    for j in range(args.wm_window):
      predicted, reward, done_logit = wm(predicted, batch["action"][:, j])
      valid = batch["mask"][:, j] * (~batch["done"][:, j]).float()
      error = ((predicted - batch["next_state"][:, j]) / wm.delta_norm.std()).square().mean(-1)
      state_loss = (error * valid).sum() / valid.sum().clamp_min(1)
      reward_error = ((reward - batch["terms"][:, j]) / wm.reward_norm.std()).square().mean(-1)
      mask = batch["mask"][:, j]
      reward_loss = (reward_error * mask).sum() / mask.sum().clamp_min(1)
      done_error = F.binary_cross_entropy_with_logits(done_logit, batch["terminated"][:, j].float(), reduction="none")
      done_loss = (done_error * mask).sum() / mask.sum().clamp_min(1)
      loss = loss + state_loss + reward_loss + done_loss
      last_error = state_loss.detach()
    optimize(loss / args.wm_window, optimizers["wm"], wm)
    return float(loss.detach() / args.wm_window), float(last_error)

  def discriminator_update() -> float:
    batch = replay.sample(args.wm_batch)
    actor, expert = batch["amp_actor"][:, 0], batch["amp_ref"][:, 0]
    inputs = torch.cat((actor, expert)).detach().requires_grad_(True)
    scores = disc(disc_norm(inputs)).squeeze(-1)
    targets = torch.cat((-torch.ones(actor.shape[0], device=device), torch.ones(expert.shape[0], device=device)))
    gradient = torch.autograd.grad(scores.sum(), inputs, create_graph=True)[0]
    penalty = gradient.norm(dim=-1).square().mean()
    loss = F.mse_loss(scores, targets) + 5 * penalty + 0.04 * disc[-1].weight.square().sum()
    optimize(loss, optimizers["disc"], disc)
    return float(loss.detach())

  def policy_update() -> tuple[float, float, float]:
    batch = replay.sample(args.pi_batch)
    imagined = batch["state"][:, 0].detach()
    hist = batch["history"].detach()
    for model in (wm, critic, disc):
      model.requires_grad_(False)
    states, next_states, rewards, continues = [], [], [], []
    alive = torch.ones(args.pi_batch, device=device)
    objective = torch.zeros_like(alive)
    for step in range(args.horizon):
      states.append(imagined)
      action = pi(hist[:, :, :PROPRIO_DIM], imagined[:, PROPRIO_DIM:PROPRIO_DIM + GOAL_DIM])
      next_state, terms, done_logit = wm(imagined, action)
      # The current action is known exactly, not a learned state channel.
      next_state = torch.cat((next_state[:, :64], action, next_state[:, 93:]), -1)
      next_hist = torch.cat((hist[:, 1:], next_state[:, None]), 1)
      style = (1 - 0.25 * (disc(disc_norm(amp_from_history(next_hist))).squeeze(-1) - 1).square()).clamp_min(0)
      reward = imagined_reward(imagined, next_state, action, terms, term_names, env.step_dt) + 0.25 * style
      continuation = 1 - done_logit.sigmoid()
      objective = objective + args.gamma**step * alive * reward
      alive = alive * continuation
      rewards.append(reward.detach())
      continues.append(continuation.detach())
      next_states.append(next_state.detach())
      imagined, hist = next_state, next_hist
    objective = objective + args.gamma**args.horizon * alive * critic(imagined).clamp(max=reward_bound / (1 - args.gamma))
    policy_loss = -objective.mean()
    grad = optimize(policy_loss, optimizers["pi"], pi)
    for model in (wm, critic, disc):
      model.requires_grad_(True)
    with torch.no_grad():
      x = torch.stack(states).detach()
      nx = torch.stack(next_states)
      value = critic(x).clamp(max=reward_bound / (1 - args.gamma))
      next_value = critic(nx).clamp(max=reward_bound / (1 - args.gamma))
      r, continuation = torch.stack(rewards), torch.stack(continues)
      delta = r + args.gamma * continuation * next_value - value
      advantage, gae = torch.empty_like(delta), torch.zeros_like(delta[0])
      for t in reversed(range(args.horizon)):
        gae = delta[t] + args.gamma * args.lam * continuation[t] * gae
        advantage[t] = gae
      target = (value + advantage).flatten()
      x = x.flatten(0, 1)
    vloss = 0.0
    for _ in range(5):
      # Uncapped regression lets overestimated values learn downwards.
      loss = F.mse_loss(critic(x), target)
      optimize(loss, optimizers["critic"], critic)
      vloss += float(loss.detach()) / 5
    return float(policy_loss.detach()), vloss, grad

  print(f"INITIALIZED state_dim={state_dim} reward_terms={len(term_names)} replay_bytes_estimate="
    f"{args.buffer_steps * args.num_envs * (2 * state_dim + 29 + len(term_names) + 2 * AMP_DIM * 10) * 4}", flush=True)
  collect(max(args.warmup_steps, args.hist + args.wm_window + 1))
  print("REPLAY_WARMUP_COMPLETE", flush=True)
  fields = ["iteration", "hours", "sim_steps", "wm_loss", "wm_error8", "pi_loss", "critic_loss", "disc_loss",
    "policy_grad", "real_reward", "termination_rate", "episode_return", "episode_seconds", "contact_bodies_mean", "terrain_level"]
  csvfile = (logdir / "train.csv").open("w", buffering=1)
  writer = csv.DictWriter(csvfile, fieldnames=fields)
  writer.writeheader()
  last_iteration = start
  for iteration in range(start + 1, args.iterations + 1):
    real_reward, term_rate = collect(args.collect_steps)
    losses = [wm_update() for _ in range(args.wm_batches)]
    disc_loss = discriminator_update()
    pi_loss, critic_loss, policy_grad = 0.0, 0.0, 0.0
    if iteration > args.wm_only_iters:
      pi_loss, critic_loss, policy_grad = policy_update()
    # Same cosine LR floor as the reference implementation.
    factor = 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(iteration, 250000) / 250000))
    for name in ("wm", "pi", "critic"):
      for group in optimizers[name].param_groups:
        group["lr"] = args.lr * factor
    last_iteration = iteration
    if iteration % args.log_every == 0 or iteration == start + 1:
      contact = env.scene["supermimic_contacts"].data.force.norm(dim=-1) > 1
      row = dict(zip(fields, [iteration, (wall_offset + time.monotonic() - started) / 3600, sim_steps,
        sum(l[0] for l in losses) / len(losses), sum(l[1] for l in losses) / len(losses), pi_loss, critic_loss,
        disc_loss, policy_grad, real_reward, term_rate,
        sum(completed_returns) / max(1, len(completed_returns)),
        sum(completed_lengths) / max(1, len(completed_lengths)), float(contact.float().sum(-1).mean()),
        float(env.scene.terrain.terrain_levels.float().mean())]))
      if not all(math.isfinite(float(v)) for v in row.values()):
        raise FloatingPointError(f"Nonfinite training metric: {row}")
      writer.writerow(row)
      print(json.dumps(row), flush=True)
    if iteration % args.ckpt_every == 0:
      save(iteration)
    if stop or (args.max_hours > 0 and time.monotonic() - started >= args.max_hours * 3600):
      break
  save(last_iteration)
  (logdir / "completed.json").write_text(json.dumps({"iteration": last_iteration, "interrupted": stop,
    "wall_time": wall_offset + time.monotonic() - started}))
  csvfile.close()
  env.close()


if __name__ == "__main__":
  main()
