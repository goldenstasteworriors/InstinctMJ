"""Probe a frozen AMP discriminator on real windows and temporal counterfactuals.

Counterfactuals are diagnostic inputs, not physically simulated trajectories.
Scores alone do not validate gait; inspect policy/reference videos separately.
"""
import argparse
import json
from pathlib import Path

import torch

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl.supermimic.style import StyleReward
from instinct_mj.rl.supermimic.parkour.mdp import AMP_HIST, AMP_TERMS
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls


def counterfactual(window, mode, generator):
    batch = len(window)
    order = torch.rand(batch, AMP_HIST, device=window.device, generator=generator).argsort(-1)
    parts = []
    for name, (lo, hi) in AMP_TERMS.items():
        x = window[:, lo * AMP_HIST:hi * AMP_HIST].reshape(batch, AMP_HIST, hi - lo)
        if mode == "static_pose":
            x = torch.zeros_like(x) if "vel" in name else x[:, -1:].expand_as(x)
        elif mode == "time_shuffled":
            x = x.gather(1, order[:, :, None].expand_as(x))
        else:
            raise ValueError(mode)
        parts.append(x.flatten(1))
    return torch.cat(parts, -1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.num_envs < 1 or args.steps <= AMP_HIST:
        raise ValueError("Require positive environments and mature AMP histories")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    sidecar = args.checkpoint.with_name(args.checkpoint.name.replace("model_", "supermimic_state_"))
    saved = torch.load(sidecar, map_location="cpu", weights_only=False)
    c = saved["cfg"]
    if saved["style"] is None:
        raise ValueError("Checkpoint has no AMP discriminator")
    task = "Instinct-Parkour-Target-Amp-SuperMimic-G1-v0"
    cfg = load_env_cfg(task)
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    cfg.auto_reset, cfg.curriculum, cfg.monitors = False, {}, {}
    env = load_env_cls(task)(cfg, device="cuda:0")
    vec = load_vecenv_cls(task)(env, policy_group="policy", critic_group="critic")
    runner = load_runner_cls(task)(vec, load_instinct_rl_cfg(task).to_dict(), device="cuda:0")
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy()
    style = StyleReward(vec.get_obs_format()["amp_policy"], "cuda:0", c["style_discriminator_kwargs"],
                        reward_coef=c["style_reward_coef"], reward_type=c["style_reward_type"])
    style.discriminator.load_state_dict(saved["style"]["discriminator"])
    style.discriminator.eval()
    del saved
    obs, _ = vec.get_observations()
    windows = {"policy": [], "reference": []}
    for _ in range(args.steps):
        obs1, _, done, infos = vec.step(policy(obs))
        valid = (env.episode_length_buf >= AMP_HIST) & ~done.bool()
        for name, group in [("policy", "amp_policy"), ("reference", "amp_reference")]:
            windows[name].append(infos["observations"][group][valid].clone())
        ids = done.nonzero().flatten()
        if len(ids):
            env.reset(env_ids=ids)
            reset_obs, _ = vec.get_observations()
            obs1 = torch.where(done.bool()[:, None], reset_obs, obs1)
        obs = obs1
    generator = torch.Generator(device="cuda:0").manual_seed(args.seed + 1)
    result = {"checkpoint": str(args.checkpoint), "seed": args.seed, "steps": args.steps,
              "protocol": "Frozen discriminator; mature pre-reset real windows, failures excluded from sampling. Original task terminations remain active. Counterfactuals are not simulator trajectories.",
              "task": task, "reward_coef": style.reward_coef, "groups": {},
              "limitations": "Temporal overlap and one policy seed; static pose is a synthetic stress test. This does not establish reward-gradient fidelity, physical feasibility, or natural gait."}
    for name, batches in windows.items():
        x = torch.cat(batches)
        if not len(x):
            raise RuntimeError("No mature windows")
        original = style.reward(x)
        variants = {"original": original}
        for mode in ("static_pose", "time_shuffled"):
            variants[mode] = style.reward(counterfactual(x, mode, generator))
        result["groups"][name] = {}
        for mode, score in variants.items():
            if not torch.isfinite(score).all():
                raise FloatingPointError("Nonfinite discriminator score")
            result["groups"][name][mode] = {
                "count": len(score), "reward_mean": float(score.mean()),
                "reward_quantiles_10_50_90": torch.quantile(score, score.new_tensor([.1, .5, .9])).tolist(),
                "mean_delta_vs_original": float((score - original).mean()),
                "fraction_above_original": float((score > original).float().mean())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
