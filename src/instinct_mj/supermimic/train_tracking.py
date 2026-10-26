"""SuperMimic training on native paired-terrain whole-body motion tracking."""

import argparse
import csv
import json
import signal
import subprocess
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from instinct_mj.envs import InstinctRlEnv
from instinct_mj.supermimic.replay import ReplayBuffer
from instinct_mj.supermimic.tracking_config import make_tracking_config
from instinct_mj.supermimic.tracking_model import (
  CONTACT, DEPTH, TORQUE, TrackingCritic, TrackingPolicy, TrackingWorldModel,
  goal, proprio, reference_at,
)
from instinct_mj.supermimic.tracking_task import TrackingTask


def optimize(loss, optimizer, model):
  if not torch.isfinite(loss):
    raise FloatingPointError("Nonfinite loss")
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
  optimizer.step()
  return float(norm)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True)
  p.add_argument("--log-dir", required=True)
  p.add_argument("--num-envs", type=int, default=512)
  p.add_argument("--iterations", type=int, default=100000)
  p.add_argument("--max-hours", type=float, default=12)
  p.add_argument("--buffer-steps", type=int, default=256)
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--wm-batches", type=int, default=4)
  p.add_argument("--wm-window", type=int, default=8)
  p.add_argument("--wm-only-iters", type=int, default=300)
  p.add_argument("--horizon", type=int, default=32)
  p.add_argument("--collect-steps", type=int, default=3)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--log-every", type=int, default=10)
  p.add_argument("--ckpt-every", type=int, default=500)
  p.add_argument("--compile", action="store_true")
  p.add_argument("--resume")
  args = p.parse_args()
  if args.horizon > 32:
    raise ValueError("Reference storage covers at most 32 imagined steps")
  torch.manual_seed(args.seed)
  torch.set_num_threads(8)
  torch.backends.cuda.matmul.allow_tf32 = True
  log = Path(args.log_dir)
  log.mkdir(parents=True, exist_ok=True)
  if (log / "last.json").exists() and not args.resume:
    raise FileExistsError("Choose a new run directory or explicitly resume")
  (log / "args.json").write_text(json.dumps(vars(args), indent=2))
  cfg = make_tracking_config(args.data, args.num_envs, args.seed)
  env = InstinctRlEnv(cfg, device="cuda:0")
  obs, _ = env.reset()
  task = TrackingTask(env)
  state, ref, ctx = task.state(obs), task.reference(), task.context()
  audit = task.audit(state)
  # FK must support gradients, not just match forward values.
  probe = state.detach().requires_grad_(True)
  probe_loss = task.reward_terms(probe, probe, probe[:, 71:100], ref[:, 0], task.limits, smooth=True).sum()
  grad = torch.autograd.grad(probe_loss, probe)[0]
  if not torch.isfinite(grad).all() or not grad.abs().any():
    raise AssertionError("Tracking reward gradient is invalid")
  audit["reward_gradient_norm"] = float(grad.norm())
  (log / "audit.json").write_text(json.dumps(audit, indent=2))
  print("AUDIT", json.dumps(audit), flush=True)
  pi, wm = TrackingPolicy().cuda(), TrackingWorldModel(ctx.shape[-1]).cuda()
  critic = TrackingCritic(ctx.shape[-1]).cuda()
  models = {"policy": pi, "world_model": wm, "critic": critic}
  opts = {k: torch.optim.Adam(v.parameters(), lr=2e-4, betas=(0.9, 0.95) if k == "policy" else (0.9, 0.999))
    for k, v in models.items()}
  start, sim_steps = 0, 0
  if args.resume:
    ckpt = torch.load(args.resume, weights_only=False, map_location="cuda:0")
    for k, v in models.items():
      v.load_state_dict(ckpt[k])
      opts[k].load_state_dict(ckpt["optimizers"][k])
    start, sim_steps = ckpt["iteration"], ckpt["sim_steps"]
  protocol = {
    "task": "Instinct-Perceptive-Shadowing-G1-v0", "algorithm": "SuperMimic paired-terrain tracking v2",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "motions": task.buffer.yaml_data, "state_dim": state.shape[-1], "context_dim": ctx.shape[-1],
    "native_rewards": task.term_names, "native_terminations": list(cfg.terminations),
    "dt": env.step_dt,
    "differences": ["physical state, analytic FK and all native tracking reward terms; no AMP",
      "full 18x32 depth CNN; world model predicts depth, contacts and torques",
      "fixed paired terrain identity and full randomized physics condition world model",
      "actor sees 10 future frames and 10 proprio frames; native PPO uses 8 proprio frames",
      "18 logged reference frames interpolate exogenously for 32-step imagination",
      "smooth contact threshold and tracking survival during imagination",
      "native early termination uses aiming reference; imagined survival uses interpolated reference",
      "imagination does not differentiate mesh ray casting; no unseen-terrain claim"],
  }
  (log / "protocol.json").write_text(json.dumps(protocol, indent=2))
  reward_fn = task.reward_terms
  if args.compile:
    reward_fn = torch.compile(reward_fn)
  replay = ReplayBuffer(args.buffer_steps, args.num_envs, "cuda:0")
  history = proprio(state, task.defaults)[:, None].repeat(1, 10, 1)
  ids = torch.arange(env.num_envs, device=env.device)
  returns, lengths = torch.zeros_like(ids, dtype=torch.float32), torch.zeros_like(ids)
  ep_returns, ep_lengths = deque(maxlen=1000), deque(maxlen=1000)
  failures = 0
  # Capture terminal physical states before auto-reset; evaluation uses the same hook.
  original_compute = env.reward_manager.compute
  capture = {}

  def compute_with_metrics(*a, **kw):
    result = original_compute(*a, **kw)
    d, r = task.robot.data, task.motion.reference_frame
    capture["root_error"] = (d.root_link_pos_w - r.base_pos_w[:, 0]).norm(dim=-1).detach()
    capture["link_error"] = (d.body_link_pos_w[:, task.links] - r.link_pos_w[:, 0]).norm(dim=-1).mean(-1).detach()
    force = env.scene["contact_forces"].data.force.norm(dim=-1)
    capture["contacts"] = (force > 1).detach()
    return result

  env.reward_manager.compute = compute_with_metrics
  metrics = {}

  @torch.no_grad()
  def collect(steps):
    nonlocal state, ref, ctx, history, sim_steps, failures, metrics
    rewards, errors, contact_sum = [], [], []
    for _ in range(steps):
      pi.obs_norm.update(history.flatten(0, 1))
      pi.goal_norm.update(goal(state, ref[:, :10], task.defaults))
      action = (pi(history, state, ref[:, :10], task.defaults) + torch.randn_like(state[:, 71:100]) * 0.1).clamp(-10, 10)
      nobs, reward, terminated, truncated, _ = env.step(action)
      done = terminated | truncated
      nxt, nref, nctx = task.state(nobs), task.reference(), task.context()
      r = reward["rewards"] if isinstance(reward, dict) else reward
      if not torch.isfinite(r).all():
        raise FloatingPointError("Nonfinite native reward")
      replay.add(state=state, next_state=nxt, action=action, reference=ref.half(),
        context=ctx, done=done, env_id=ids)
      valid = ~done
      if valid.any():
        wm.input_norm.update(torch.cat((state[valid], action[valid], ctx[valid]), -1))
        wm.delta_norm.update(nxt[valid] - state[valid])
      critic.norm.update(torch.cat((state, ref[:, 0, :71], ctx), -1))
      # Verify the native action penalty on the exact executed action.
      expected = -0.1 * (action - state[:, 71:100]).square().sum(-1) * env.step_dt
      actual = env.reward_manager._termwise_reward_buf["rewards"]["action_rate_l2"]
      if not torch.allclose(expected, actual, atol=2e-5, rtol=1e-4):
        raise AssertionError("Action reward mismatch")
      returns.add_(r); lengths.add_(1)
      if done.any():
        ep_returns.extend(returns[done].tolist())
        ep_lengths.extend((lengths[done] * env.step_dt).tolist())
        returns[done] = 0; lengths[done] = 0
      failures += int(terminated.sum())
      fresh = proprio(nxt, task.defaults)
      history = torch.cat((history[:, 1:], fresh[:, None]), 1)
      history[done] = fresh[done, None].expand(-1, 10, -1)
      state, ref, ctx = nxt, nref, nctx
      sim_steps += env.num_envs
      rewards.append(float(r.mean())); errors.append(float(capture["root_error"].mean()))
      contact_sum.append(capture["contacts"].float().mean(0))
    metrics = {"real_reward": sum(rewards) / steps, "root_error_m": sum(errors) / steps,
      "contacts": torch.stack(contact_sum).mean(0)}

  def wm_update():
    b = replay.sample(args.batch, args.wm_window)
    predicted, loss = b["state"][:, 0], torch.zeros((), device=env.device)
    contact_error = torch.zeros_like(loss)
    for j in range(args.wm_window):
      predicted = wm(predicted, b["action"][:, j], b["context"][:, j])
      mask = b["mask"][:, j] * (~b["done"][:, j]).float()
      err = ((predicted - b["next_state"][:, j]) / wm.delta_norm.std()).square()
      per_sample = err[:, :71].mean(-1) + 0.5 * err[:, DEPTH].mean(-1) + 0.2 * err[:, CONTACT].mean(-1) + 0.2 * err[:, TORQUE].mean(-1)
      loss = loss + (per_sample * mask).sum() / mask.sum().clamp_min(1)
      contact_error = contact_error + (err[:, CONTACT].mean(-1) * mask).sum() / mask.sum().clamp_min(1)
    optimize(loss / args.wm_window, opts["world_model"], wm)
    return float(loss.detach() / args.wm_window), float(contact_error.detach() / args.wm_window)

  def policy_update():
    b = replay.sample(args.batch)
    s, ctx = b["state"][:, 0], b["context"][:, 0]
    env_ids = b["env_id"][:, 0]
    defaults, limits = task.defaults[env_ids], task.limits[env_ids]
    hist = proprio(b["history"], defaults[:, None])
    refs = reference_at(b["reference"][:, 0], torch.arange(args.horizon + 46, device=env.device))
    for model in (wm, critic):
      model.requires_grad_(False)
    alive, objective = torch.ones(args.batch, device=env.device), torch.zeros(args.batch, device=env.device)
    xs, ns, rs, cs, rfs = [], [], [], [], []
    for j in range(args.horizon):
      future = refs[:, j + torch.arange(10, device=env.device) * 5]
      a = pi(hist, s, future, defaults)
      nxt = wm(s, a, ctx)
      terms, cont = reward_fn(s, nxt, a, refs[:, j + 1], limits, smooth=True, with_continuation=True)
      cont = cont * refs[:, j + 1, 253].clamp(0, 1)
      reward = terms.sum(-1)
      objective = objective + 0.99**j * alive * reward
      alive = alive * cont
      xs.append(s.detach()); ns.append(nxt.detach()); rs.append(reward.detach()); cs.append(cont.detach())
      rfs.append(refs[:, j, :71])
      hist = torch.cat((hist[:, 1:], proprio(nxt, defaults)[:, None]), 1)
      s = nxt
    objective = objective + 0.99**args.horizon * alive * critic(s, refs[:, args.horizon], ctx).clamp(max=10)
    policy_loss = -objective.mean()
    grad = optimize(policy_loss, opts["policy"], pi)
    for model in (wm, critic):
      model.requires_grad_(True)
    with torch.no_grad():
      x, nx, r, c, rf = torch.stack(xs), torch.stack(ns), torch.stack(rs), torch.stack(cs), torch.stack(rfs)
      contexts = ctx[None].expand(args.horizon, -1, -1)
      v = critic(x, rf, contexts).clamp(max=10)
      next_refs = refs[:, 1:args.horizon + 1, :71].transpose(0, 1)
      nv = critic(nx, next_refs, contexts).clamp(max=10)
      delta = r + 0.99 * c * nv - v
      gae = torch.zeros_like(alive)
      advantages = torch.empty_like(r)
      for j in reversed(range(args.horizon)):
        gae = delta[j] + 0.99 * 0.95 * c[j] * gae
        advantages[j] = gae
      targets = (v + advantages).flatten()
    critic_loss = 0
    for _ in range(5):
      loss = F.mse_loss(critic(x.flatten(0, 1), rf.flatten(0, 1), contexts.flatten(0, 1)), targets)
      optimize(loss, opts["critic"], critic)
      critic_loss += float(loss.detach()) / 5
    return float(policy_loss.detach()), critic_loss, grad

  stop = False
  def request_stop(*_):
    nonlocal stop
    stop = True
  signal.signal(signal.SIGTERM, request_stop)
  signal.signal(signal.SIGINT, request_stop)
  began = time.monotonic()

  def save(iteration):
    path = log / f"ckpt_{iteration:06d}.pt"
    torch.save({**{k: v.state_dict() for k, v in models.items()},
      "optimizers": {k: v.state_dict() for k, v in opts.items()}, "args": vars(args),
      "iteration": iteration, "sim_steps": sim_steps, "context_dim": ctx.shape[-1],
      "protocol": protocol}, path)
    (log / "last.json").write_text(json.dumps({"checkpoint": str(path), "iteration": iteration,
      "wall_time": time.monotonic() - began, "sim_steps": sim_steps}, indent=2))
    print("CHECKPOINT", str(path), flush=True)

  fields = ["iteration", "hours", "sim_steps", "wm_loss", "contact_model_loss", "policy_loss", "critic_loss", "policy_grad",
    "real_reward", "root_error_m", "episode_return", "episode_seconds", "failures", "nonfoot_contact_events_per_step"]
  csv_path = log / "train.csv"
  stream = csv_path.open("a", buffering=1)
  writer = csv.DictWriter(stream, fieldnames=fields)
  if stream.tell() == 0:
    writer.writeheader()
  print("INITIALIZED", json.dumps({"envs": args.num_envs, "motions": len(task.buffer.yaml_data["motion_files"]),
    "terrains": task.num_terrains, "context": ctx.shape[-1]}), flush=True)
  collect(max(32, args.wm_window + 11))
  feet = task.robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"])[0]
  nonfoot = [i for i in range(40) if i not in feet]
  iteration = start
  for iteration in range(start + 1, args.iterations + 1):
    collect(args.collect_steps)
    losses = [wm_update() for _ in range(args.wm_batches)]
    ploss, vloss, pgrad = (0, 0, 0) if iteration <= args.wm_only_iters else policy_update()
    if iteration % args.log_every == 0 or iteration == start + 1:
      row = dict(zip(fields, [iteration, (time.monotonic() - began) / 3600, sim_steps,
        sum(x[0] for x in losses) / len(losses), sum(x[1] for x in losses) / len(losses), ploss, vloss, pgrad,
        metrics["real_reward"], metrics["root_error_m"], sum(ep_returns) / max(1, len(ep_returns)),
        sum(ep_lengths) / max(1, len(ep_lengths)), failures, float(metrics["contacts"][nonfoot].sum())]))
      writer.writerow(row)
      print("TRAIN", json.dumps(row), flush=True)
    if iteration % args.ckpt_every == 0 or iteration == args.wm_only_iters:
      save(iteration)
    if stop or (args.max_hours > 0 and time.monotonic() - began >= args.max_hours * 3600):
      break
  save(iteration)
  (log / "completed.json").write_text(json.dumps({"iteration": iteration, "interrupted": stop,
    "wall_time": time.monotonic() - began}))
  stream.close()
  env.close()


if __name__ == "__main__":
  main()
