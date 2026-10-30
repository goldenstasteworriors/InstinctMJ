"""Decompose a saved SuperMimic policy gradient without taking an optimizer step.

Uses the production imagination, rewards, survival and critic bootstrap. Reports
gradient directions, not evidence that those directions improve the simulator.
Reference statistics are from the actual environment AMP FIFO, not raw files.
"""

import argparse
import json
from pathlib import Path

import torch

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl.supermimic.parkour.mdp import AMP_HIST, AMP_TERMS
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.num_envs, args.batches) < 1:
        raise ValueError("Positive batch sizes required")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    sidecar = args.checkpoint.with_name(args.checkpoint.name.replace("model_", "supermimic_state_"))
    saved = torch.load(sidecar, map_location="cpu", weights_only=False)
    saved_cfg, mu = saved["cfg"], float(saved["fr_mean"].item())
    del saved
    cfg = load_env_cfg(args.task)
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    cfg.auto_reset, cfg.curriculum, cfg.monitors = False, {}, {}
    cfg.events["physics_material"].params.update(static_friction_range=(mu, mu), dynamic_friction_range=(mu, mu))
    env = load_env_cls(args.task)(cfg, device="cuda:0")
    vec = load_vecenv_cls(args.task)(env, policy_group="policy", critic_group="critic")
    agent = load_instinct_rl_cfg(args.task).to_dict()
    policy_class = agent["policy"]["class_name"]
    agent.update(saved_cfg)
    agent["policy"] = dict(saved_cfg["policy"], class_name=policy_class)
    agent.update(compile=False, compile_policy=False, buffer_steps=128, pi_batch=args.num_envs)
    runner = load_runner_cls(args.task)(vec, agent, device="cuda:0")
    runner.load(str(args.checkpoint))
    runner.setup_training()
    runner.eval_mode()
    stats = {key: [] for key in ("r", "r_gap", "ep_len", "term", "done")}
    runner.collect(128, stats)
    # Collection updates observation/WM normalization statistics. Restore them
    # and all saved weights before measuring the checkpoint's gradients.
    runner.load(str(args.checkpoint))
    runner._load_sidecar(str(args.checkpoint))
    frozen = (runner.wm, runner.critic, runner.fail_head)
    if runner.style is not None:
        frozen += (runner.style.discriminator,)
    for module in frozen:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    parameters = [p for p in runner.ac.parameters() if p.requires_grad]

    def gradient(objective):
        grads = torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True) if objective.requires_grad else [None] * len(parameters)
        return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(parameters, grads)]).detach()

    rows = []
    for batch in range(args.batches):
        s, h = runner.sample_valid_starts(args.num_envs)
        im = runner._imagine(s, h)
        horizon, B = runner.horizon, args.num_envs
        flat = lambda xs: torch.stack(xs).flatten(0, 1)
        rep = lambda x: x.unsqueeze(0).expand(horizon, *x.shape).flatten(0, 1)
        states = tuple(flat([st[i] for st in im["Ss"]]) for i in range(6))
        previous = tuple(flat([st[i] for st in im["S_prev"]]) for i in range(6))
        task, done, terms = runner.reward_term(previous, states, flat(im["As"]), flat(im["Aps"]), rep(im["cmd"]), flat(im["Ts"]), flat(im["Ls"]))
        task = task + runner.mdp.volume_points_penetration(states)
        style = runner.style.reward(runner.mdp.amp_window(flat(im["Amps"]))) if runner.style else torch.zeros_like(task)
        if agent["fail_head"] and runner.it >= runner.fail_head_start_it + agent["fail_head_warmup_iters"]:
            failure = runner.fail_head(runner.critic_input(states, rep(im["cmd"]), flat(im["Ts"]), rep(im["xi_n"]))).sigmoid().clamp(max=agent["fail_clip"])
            done = 1 - (1 - done) * (1 - failure)
        alive = torch.cumprod(1 - done.view(horizon, B), 0)
        survival = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        discount = agent["gamma"] ** torch.arange(horizon, device=task.device).float()[:, None]
        def cumulative(reward):
            return (discount * survival * reward.view(horizon, B)).sum(0).mean()
        tracking = terms["track_lin_vel_xy_exp"] + terms["track_ang_vel_z_exp"]
        parts = dict(velocity_tracking=cumulative(tracking), other_task=cumulative(task - tracking), amp=cumulative(style))
        parts["critic_bootstrap"] = task.new_zeros(()) if agent["no_critic"] else (
            agent["gamma"] ** horizon * alive[-1] * runner.critic(runner.critic_input(im["S"], im["cmd"], im["timers"], im["xi_n"]))
        ).mean()
        total = gradient(sum(parts.values()))
        gradients = {name: gradient(value) for name, value in parts.items()}
        if not torch.isfinite(total).all() or any(not torch.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError("Nonfinite policy gradient")
        total_sq = total.square().sum().clamp(min=1e-20)
        row = dict(batch=batch, total_gradient_norm=float(total.norm()), objective=float(sum(parts.values()).detach()),
            component_sum_relative_error=float((sum(gradients.values()) - total).norm() / total.norm().clamp(min=1e-10)),
            components={name: dict(objective=float(parts[name].detach()), gradient_norm=float(g.norm()),
                cosine_with_total=float((g * total).sum() / (g.norm() * total.norm()).clamp(min=1e-20)),
                projection_fraction=float((g * total).sum() / total_sq)) for name, g in gradients.items()})
        rows.append(row)
        print(json.dumps(row), flush=True)
        del im, parts, gradients, total
    reference = None
    if runner.style is not None:
        lo, hi = AMP_TERMS["base_lin_vel"]
        windows = runner.amp_ref[:runner.amp_ref_size, :, lo * AMP_HIST:hi * AMP_HIST].float().reshape(-1, AMP_HIST, 3)
        velocity = windows.mean(1)
        reference = dict(windows=len(velocity), forward_quantiles_mps=torch.quantile(velocity[:, 0], velocity.new_tensor([.1, .5, .9])).tolist(),
            backward_fraction=float((velocity[:, 0] < -.1).float().mean()),
            forward_fraction=float((velocity[:, 0] > .1).float().mean()),
            lateral_fraction=float((velocity[:, 1].abs() > .2).float().mean()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(protocol=__doc__, checkpoint=str(args.checkpoint), seed=args.seed,
        friction=mu, action_noise=agent["action_noise"], horizon=runner.horizon,
        wm_direct_terrain_scan=agent.get("wm_use_privileged_scan", True),
        reference_statistics=reference, records=rows), indent=2))
    env.close()


if __name__ == "__main__":
    main()
