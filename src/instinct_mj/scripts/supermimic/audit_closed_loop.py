"""Paired closed-loop simulator/model continuations from the same real state.

Both branches use the same learned actor. The model branch predicts future visual
latents; it does not replay the simulator's actions. Independent future observation
noise is part of this deployment test. Censor errors after real failure, truncation
or command resampling; keep survival calibration on the valid initial cohort.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl.supermimic.parkour.mdp import AMP_HIST, HIST, PROPRIO_TERMS
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num-envs", default=128, type=int)
    p.add_argument("--seed", default=1234, type=int)
    p.add_argument("--windows", default=16, type=int)
    p.add_argument("--horizon", default=32, type=int)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    cfg = load_env_cfg(args.task)
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    cfg.auto_reset, cfg.curriculum, cfg.monitors = False, {}, {}
    env = load_env_cls(args.task)(cfg, device="cuda:0")
    vec = load_vecenv_cls(args.task)(env, policy_group="policy", critic_group="critic")
    agent = load_instinct_rl_cfg(args.task).to_dict()
    agent.update(compile=False, compile_policy=False, buffer_steps=64)
    runner = load_runner_cls(args.task)(vec, agent, device="cuda:0")
    runner.load(str(args.checkpoint))
    runner.setup_training()
    runner.eval_mode()
    torch.manual_seed(args.seed)
    mdp, reader = runner.mdp, runner.reader
    policy = runner.get_inference_policy()
    obs, _ = vec.get_observations()
    amp_hist = mdp.amp_frame(reader.state()).unsqueeze(1).repeat(1, AMP_HIST, 1)
    rows = []

    def real_step(action):
        nonlocal obs, amp_hist
        obs1, reward, done, _ = vec.step(action)
        state = reader.state()
        contact, terminated = reader.contact(), env.reset_terminated.clone()
        amp_hist = torch.cat((amp_hist[:, 1:], mdp.amp_frame(state).unsqueeze(1)), 1)
        style = runner.style.reward(mdp.amp_window(amp_hist)) if runner.style else torch.zeros_like(reward.sum(-1))
        # Read all pre-reset values before resetting finished environments.
        obs = obs1
        ids = done.nonzero().flatten()
        if len(ids):
            env.reset(env_ids=ids)
            reset_obs, _ = vec.get_observations()
            obs = torch.where(done.bool()[:, None], reset_obs, obs1)
            amp_hist[ids] = mdp.amp_frame(reader.state())[ids, None].expand(-1, AMP_HIST, -1)
        return state, contact, terminated, done.bool(), reward.sum(-1), style

    for _ in range(AMP_HIST + 8):
        real_step(policy(obs))
    for window in range(args.windows):
        S = reader.state()
        cmd = reader.command_params().clone()
        a_prev, timers = reader.last_action(), reader.feet_timers()
        xi = runner.episode_context(reader.actuator_lags(), runner.fr_n)
        z = runner.encode_depth(obs[:, runner.depth_slice].view(args.num_envs, *runner.perception_shape))
        # Invert policy_obs exactly, retaining real sensor noise and reset history.
        parts, offset = [], 0
        for lo, hi in PROPRIO_TERMS.values():
            size = (hi - lo) * HIST
            parts.append(obs[:, offset:offset + size].reshape(args.num_envs, HIST, hi - lo))
            offset += size
        o_hist = torch.cat(parts, -1)
        assert torch.allclose(mdp.policy_obs(o_hist, obs[:, runner.depth_slice].view(args.num_envs, *runner.perception_shape)), obs)
        amp_im = amp_hist.clone()
        eligible = (reader.command_time_left() > (args.horizon + 2) * env.step_dt) & (env.episode_length_buf >= AMP_HIST)
        eligible &= env.episode_length_buf + args.horizon < env.max_episode_length
        alive = eligible.clone()
        predicted_survival = torch.ones(args.num_envs, device=env.device)
        real_survival = torch.ones_like(predicted_survival)
        sim_return, model_return = torch.zeros_like(real_survival), torch.zeros_like(real_survival)
        scan = runner.wm_scan(S)
        for step in range(1, args.horizon + 1):
            S0, ap = S, a_prev
            a_prev, S, z, logits, timers, o_hist, amp_im, scan = runner.imag_core(S, z, o_hist, amp_im, timers, a_prev, cmd, xi, scan)
            model_r, d, _ = runner.reward_term(S0, S, a_prev, ap, cmd, timers, logits)
            model_style = runner.style.reward(mdp.amp_window(amp_im)) if runner.style else torch.zeros_like(model_r)
            if agent["fail_head"]:
                pf = runner.fail_head(runner.critic_input(S, cmd, timers, xi)).sigmoid().clamp(max=agent["fail_clip"])
                d = 1 - (1 - d) * (1 - pf.squeeze(-1))
            predicted_survival *= 1 - d
            real_a = policy(obs)
            Sr, contact, terminated, done, real_r, real_style = real_step(real_a)
            # Include the first failing transition in state errors, then censor resets.
            valid = alive.clone()
            real_survival[terminated & alive] = 0
            # Non-time-limit truncations cannot be interpreted as a failure label.
            eligible &= ~(done & ~terminated & alive)
            sim_return += (real_r + real_style) * alive
            model_return += (model_r + model_style) * predicted_survival
            if step in (1, 8, 16, 32) or step == args.horizon:
                n = int(valid.sum())
                def mean(x):
                    return float(x[valid].mean()) if n else None
                rows.append(dict(window=window, step=step, valid=n, cohort=int(eligible.sum()),
                    position_rmse_m=math.sqrt(mean((S[0] - Sr[0]).square().mean(-1))) if n else None,
                    velocity_rmse_mps=math.sqrt(mean((S[2] - Sr[2]).square().mean(-1))) if n else None,
                    action_rmse=math.sqrt(mean((a_prev - real_a).square().mean(-1))) if n else None,
                    contact_brier=mean((logits.sigmoid() - contact).square().mean(-1)),
                    style_real=mean(real_style), style_model=mean(model_style),
                    task_reward_real=mean(real_r), task_reward_model=mean(model_r),
                    survival_real=float(real_survival[eligible].mean()) if eligible.any() else None,
                    survival_model=float(predicted_survival[eligible].mean()) if eligible.any() else None,
                    survival_brier=float((predicted_survival[eligible] - real_survival[eligible]).square().mean()) if eligible.any() else None))
            alive &= ~done
        print(json.dumps(rows[-1]), flush=True)
        for _ in range(AMP_HIST):
            real_step(policy(obs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "task": args.task,
        "protocol": __doc__, "seed": args.seed, "records": rows}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
