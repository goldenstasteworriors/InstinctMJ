"""First-episode locomotion evaluation with training terminations and pre-reset metrics.

Reports survival separately from locomotion quality: a 20 s timeout is not proof
of reaching a target or traversing obstacles. Uses the training task with a fresh
seed and its domain randomization, camera noise and initial terrain levels 0..5.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from instinct_rl.runners import OnPolicyRunner

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl.supermimic.parkour.sim import ParkourSimReader
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls

TASK = "Instinct-Parkour-Target-Amp-SuperMimic-G1-v0"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--perception-intervention", choices=("intact", "shuffled", "mean"), default="intact")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-envs", default=128, type=int)
    parser.add_argument("--seconds", default=20.0, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--video", type=Path, help="Optional four-view MP4; first episodes freeze at termination.")
    parser.add_argument("--video-envs", nargs=4, type=int, default=[0, 32, 64, 96])
    parser.add_argument("--video-label", default="SuperMimic locomotion")
    args = parser.parse_args()
    if args.num_envs < 2 or args.seconds <= 0:
        raise ValueError("Evaluation requires at least two environments and positive seconds")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    cfg = load_env_cfg(args.task)
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    cfg.auto_reset, cfg.monitors, cfg.curriculum = False, {}, {}
    cfg.episode_length_s = args.seconds
    env = load_env_cls(args.task)(cfg, device="cuda:0")
    vec = load_vecenv_cls(args.task)(env, policy_group="policy", critic_group="critic")
    runner_cls = load_runner_cls(args.task) or OnPolicyRunner
    runner = runner_cls(vec, load_instinct_rl_cfg(args.task).to_dict(), device="cuda:0")
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy()
    reader = ParkourSimReader(env)
    obs, _ = vec.get_observations()
    n, dev = args.num_envs, env.device
    active = torch.ones(n, dtype=torch.bool, device=dev)
    steps = torch.zeros(n, device=dev)
    returns, lin_sq, yaw_sq, travel = (steps.clone() for _ in range(4))
    levels = env.scene.terrain.terrain_levels.clone()
    kinds = env.scene.terrain.terrain_types.clone()
    starts = reader.state()[0].clone()
    end_pos = starts.clone()
    result, reasons = ["step_limit"] * n, [[] for _ in range(n)]
    contact_sum = torch.zeros(reader.num_contacts, device=dev)
    video = None
    if args.video is not None:
        from instinct_mj.scripts.supermimic.parkour_video import ParkourVideo

        video = ParkourVideo(env, args.video, args.video_envs, args.video_label, levels, kinds)
    for step in range(1, math.ceil(args.seconds / env.step_dt) + 2):
        command = reader.command.command.clone()
        before = reader.state()[0]
        policy_obs = obs
        if args.perception_intervention != "intact":
            policy_obs = obs.clone()
            visual = obs[:, runner.depth_slice]
            policy_obs[:, runner.depth_slice] = visual.roll(1, 0) if args.perception_intervention == "shuffled" else visual.mean(0, keepdim=True)
        action = policy(policy_obs)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite evaluation action")
        obs1, reward, dones, _ = vec.step(action)
        done = dones.bool()
        p, R, v, w, _, _ = reader.state()
        if not all(torch.isfinite(x[active]).all() for x in (p, R, v, w, reward)):
            raise FloatingPointError("Nonfinite evaluation state or reward")
        vb = (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
        wb = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        steps += active
        returns += reward.sum(-1) * active
        lin_sq += (vb[:, :2] - command[:, :2]).square().sum(-1) * active
        yaw_sq += (wb[:, 2] - command[:, 2]).square() * active
        travel += (p[:, :2] - before[:, :2]).norm(dim=-1) * active
        end_pos[active] = p[active]
        contact_sum += (reader.contact() * active[:, None]).sum(0)
        for i in torch.where(done & active)[0].tolist():
            terms = [name for name in env.termination_manager.active_terms if env.termination_manager.get_term(name)[i]]
            reasons[i] = terms
            result[i] = "failure" if env.reset_terminated[i] else ("survived_horizon" if "time_out" in terms else "other_truncation")
        if video is not None:
            video.capture(step, active, done, result, reasons)
        active &= ~done
        if not active.any():
            break
        ids = torch.where(done)[0]
        if ids.numel():
            env.reset(env_ids=ids)
            reset_obs, _ = vec.get_observations()
            obs = torch.where(done[:, None], reset_obs, obs1)
        else:
            obs = obs1
    records = [{"terrain_level": int(levels[i]), "terrain_type": int(kinds[i]), "result": result[i],
                "termination_terms": reasons[i], "seconds": float(steps[i] * env.step_dt), "return": float(returns[i]),
                "linear_velocity_rmse_mps": float((lin_sq[i] / steps[i].clamp_min(1)).sqrt()),
                "yaw_velocity_rmse_radps": float((yaw_sq[i] / steps[i].clamp_min(1)).sqrt()),
                "travel_xy_m": float(travel[i]), "displacement_xy_m": float((end_pos[i, :2] - starts[i, :2]).norm())}
               for i in range(n)]
    summary = {"checkpoint": str(args.checkpoint), "iteration": runner.it, "seed": args.seed, "episodes": n,
               "task": args.task, "perception_intervention": args.perception_intervention,
               "protocol": "First episode; training randomization/terminations; pre-reset metrics; no curriculum updates",
               "survived_horizon": result.count("survived_horizon"), "failed": result.count("failure"),
               "censored": sum(x not in ("failure", "survived_horizon") for x in result),
               "mean_seconds": float(steps.mean() * env.step_dt),
               "linear_velocity_rmse_mps": float((lin_sq.sum() / steps.sum()).sqrt()),
               "yaw_velocity_rmse_radps": float((yaw_sq.sum() / steps.sum()).sqrt()),
               "contact_fraction_feet_torso_undesired": (contact_sum / steps.sum()).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**summary, "records": records}, indent=2))
    print(json.dumps(summary), flush=True)
    if video is not None:
        video.close()
    env.close()


if __name__ == "__main__":
    main()
