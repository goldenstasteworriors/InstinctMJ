"""Compare WM action derivatives with central differences in the real simulator.

This is a local one-control-step dynamics test, not a policy-return gradient test.
Friction is fixed to the checkpoint mean and actuator delays are zero (a trained
delay value). Each perturbation restores integration state in the same world.
No reset/termination is applied inside the physics probe; warmup uses task resets.
Repeated unperturbed steps must agree before any derivative result is accepted.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import warp as wp

import instinct_mj.tasks  # noqa: F401
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epsilon", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.num_envs, args.windows) < 1 or min(args.epsilon) <= 0:
        raise ValueError("Positive sizes and perturbations required")
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
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_min_lag = actuator.delay_max_lag = 0
    env = load_env_cls(args.task)(cfg, device="cuda:0")
    vec = load_vecenv_cls(args.task)(env, policy_group="policy", critic_group="critic")
    agent = load_instinct_rl_cfg(args.task).to_dict()
    policy_class = agent["policy"]["class_name"]
    agent.update(saved_cfg)
    agent["policy"] = dict(saved_cfg["policy"], class_name=policy_class)
    agent.update(compile=False, compile_policy=False, buffer_steps=64)
    runner = load_runner_cls(args.task)(vec, agent, device="cuda:0")
    runner.load(str(args.checkpoint))
    runner.setup_training()
    runner.eval_mode()
    reader, wm = runner.reader, runner.wm
    policy = runner.get_inference_policy()
    for parameter in wm.parameters():
        parameter.requires_grad_(False)
    def arrays(value, prefix=""):
        for field in dataclasses.fields(value):
            child = getattr(value, field.name)
            name = prefix + field.name
            if isinstance(child, wp.array):
                yield name, child
            elif dataclasses.is_dataclass(child):
                yield from arrays(child, name + ".")
    # Include collision and solver workspace: restoring only qpos/qvel and the
    # warm start did not pass the numerical repeatability gate on this backend.
    data_arrays = dict(arrays(env.sim.data.struct))
    if not {"qpos", "qvel", "qacc_warmstart"}.issubset(data_arrays):
        raise RuntimeError("Incomplete simulator state inventory")
    snapshot_bytes = sum(value.capacity for value in data_arrays.values())
    if snapshot_bytes > 4 * 1024**3:
        raise RuntimeError(f"Diagnostic snapshot exceeds 4 GiB: {snapshot_bytes}")
    snapshot = {name: wp.clone(value) for name, value in data_arrays.items()}
    records, repeatability = [], []
    for window in range(args.windows):
        env.reset()
        obs, _ = vec.get_observations()
        with torch.no_grad():
            for _ in range(64 + window * 16):
                obs1, _, done, _ = vec.step(policy(obs))
                ids = done.nonzero().flatten()
                if len(ids):
                    env.reset(env_ids=ids)
                    reset_obs, _ = vec.get_observations()
                    obs1 = torch.where(done.bool()[:, None], reset_obs, obs1)
                obs = obs1
            S = reader.state()
            a = policy(obs).clamp(-20, 20)
            previous = reader.last_action()
            lags = reader.actuator_lags()
            if not torch.equal(lags, torch.zeros_like(lags)):
                raise RuntimeError("Probe requires zero actuator delay")
            if not torch.allclose(reader.friction(), torch.full_like(reader.friction(), mu), atol=1e-5):
                raise RuntimeError("Probe friction mismatch")
            xi = runner.episode_context(lags, runner.fr_n)
            context = runner.wm_context(xi, previous)
            z = runner.encode_depth(obs[:, runner.depth_slice].view(args.num_envs, *runner.perception_shape))
            scan = runner.wm_scan(S)
            initial_contact = reader.contact()[:, :2].clone()
            torch.cuda.synchronize()
            for name, value in data_arrays.items():
                wp.copy(snapshot[name], value)
            wp.synchronize()
            direction = torch.randn_like(a)
            direction /= direction.norm(dim=-1, keepdim=True)
        # Express outputs in the fixed initial root frame, avoiding derivative
        # terms from a moving output coordinate system.
        def velocity(state):
            return (S[1].transpose(-1, -2) @ state[2].unsqueeze(-1)).squeeze(-1)

        @torch.no_grad()
        def physical(action):
            torch.cuda.synchronize()
            for name, value in data_arrays.items():
                wp.copy(value, snapshot[name])
            wp.synchronize()
            env.action_manager.process_action(action)
            for _ in range(env.cfg.decimation):
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step()
                env.scene.update(dt=env.physics_dt)
            return velocity(reader.state()).clone()

        with torch.enable_grad():
            action = a.detach().requires_grad_(True)
            model_state, _, _ = wm.step(S, action, context, scan, z)
            model_v = velocity(model_state)
            derivative = torch.stack([
                (torch.autograd.grad(model_v[:, axis].sum(), action, retain_graph=axis < 2)[0] * direction).sum(-1)
                for axis in range(3)], -1).detach()
        base = physical(a)
        for epsilon in args.epsilon:
            if (a.abs() + epsilon * direction.abs()).max() >= 20:
                raise RuntimeError("Action clipping invalidates central difference")
            plus, minus = physical(a + epsilon * direction), physical(a - epsilon * direction)
            real_derivative = (plus - minus) / (2 * epsilon)
            with torch.no_grad():
                model_plus = velocity(wm.step(S, a + epsilon * direction, context, scan, z)[0])
                model_minus = velocity(wm.step(S, a - epsilon * direction, context, scan, z)[0])
                model_fd = (model_plus - model_minus) / (2 * epsilon)
            records.append(dict(window=window, epsilon=epsilon,
                real_derivative=real_derivative.cpu().tolist(), model_derivative=derivative.cpu().tolist(),
                model_central_difference=model_fd.cpu().tolist(), initial_feet_contact=initial_contact.cpu().tolist(),
                real_velocity=base.cpu().tolist(), model_velocity=model_v.detach().cpu().tolist()))
        repeated = physical(a)
        error = float((repeated - base).abs().max())
        repeatability.append(error)
        if not torch.isfinite(repeated).all() or error > 1e-5:
            raise RuntimeError(f"Restored simulator is not repeatable: velocity max error {error}")
        print(json.dumps(dict(window=window, repeatability_velocity_absmax=error)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(protocol=__doc__, checkpoint=str(args.checkpoint),
        task=args.task, seed=args.seed, friction=mu, actuator_delay_steps=0,
        dt=env.step_dt, restored_fields=list(data_arrays), snapshot_bytes=snapshot_bytes,
        repeatability_velocity_absmax=repeatability,
        records=records), indent=2))
    env.close()


if __name__ == "__main__":
    main()
