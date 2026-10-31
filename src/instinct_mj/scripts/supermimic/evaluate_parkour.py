"""First-episode locomotion evaluation with training terminations and pre-reset metrics.

Reports survival separately from locomotion quality: a 20 s timeout is not proof
of reaching a target or traversing obstacles. Uses the training task with a fresh
seed and its domain randomization, camera noise and initial terrain levels 0..5.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from instinct_rl.runners import OnPolicyRunner
from mujoco_warp._src.types import OverflowType

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
    parser.add_argument("--command-mode", choices=("task", "forward"), default="task")
    parser.add_argument("--terrain-probe", choices=("task", "flat"), default="task",
                        help="Separate gait-only probe: large flat terrain without walls; never a parkour success test")
    parser.add_argument("--forward-speed", default=0.6, type=float)
    parser.add_argument("--friction-coefficient", type=float,
                        help="Optional common simulator friction for controlled cross-policy gait comparisons")
    parser.add_argument("--initial-terrain-level", type=int,
                        help="Optional maximum initial level, e.g. 0 for a separate easy-terrain gait probe")
    parser.add_argument("--video", type=Path, help="Optional four-view MP4; first episodes freeze at termination.")
    parser.add_argument("--video-envs", nargs=4, type=int, default=[0, 32, 64, 96])
    parser.add_argument("--video-label", default="SuperMimic locomotion")
    parser.add_argument("--trajectory", type=Path, help="Optional NPZ of post-step, pre-reset gait data; valid mask includes first failure.")
    args = parser.parse_args()
    if args.num_envs < 2 or args.seconds <= 0:
        raise ValueError("Evaluation requires at least two environments and positive seconds")
    if args.output.exists() or (args.video is not None and args.video.exists()):
        raise FileExistsError("Refusing to overwrite an existing evaluation artifact")
    if args.trajectory is not None and (args.trajectory.suffix != ".npz" or args.trajectory.exists()):
        raise ValueError("Trajectory must be a new .npz file")
    if args.forward_speed <= 0 or (args.initial_terrain_level is not None and args.initial_terrain_level < 0):
        raise ValueError("Invalid forward speed or terrain level")
    if args.friction_coefficient is not None and args.friction_coefficient <= 0:
        raise ValueError("Controlled friction coefficient must be positive")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    cfg = load_env_cfg(args.task)
    cfg.seed, cfg.scene.num_envs = args.seed, args.num_envs
    cfg.auto_reset, cfg.monitors, cfg.curriculum = False, {}, {}
    cfg.episode_length_s = args.seconds
    if args.terrain_probe == "flat":
        if args.command_mode != "forward":
            raise ValueError("Flat gait probe requires an explicit forward command")
        generator = cfg.scene.terrain.terrain_generator
        flat = copy.deepcopy(generator.sub_terrains["perlin_rough"])
        flat.noise_scale, flat.wall_prob, flat.proportion = [0.0, 0.0], [0.0] * 4, 1.0
        generator.sub_terrains = {"perlin_rough": flat}
        generator.num_rows = generator.num_cols = 1
        extent = max(30.0, 2 * (args.forward_speed * args.seconds + 3.0))
        generator.size = (extent, extent)
        cfg.scene.terrain.max_init_terrain_level = 0
        command_cfg = cfg.commands["base_velocity"]
        command_cfg.velocity_ranges = {"perlin_rough": command_cfg.velocity_ranges["perlin_rough"]}
    if args.friction_coefficient is not None:
        mu = args.friction_coefficient
        cfg.events["physics_material"].params.update(static_friction_range=(mu, mu), dynamic_friction_range=(mu, mu))
    if args.initial_terrain_level is not None:
        cfg.scene.terrain.max_init_terrain_level = args.initial_terrain_level
    if args.command_mode == "forward":
        command_cfg = cfg.commands["base_velocity"]
        command_cfg.random_velocity_terrain = list(cfg.scene.terrain.terrain_generator.sub_terrains)
        command_cfg.rel_standing_envs = 0.0
        command_cfg.ranges.lin_vel_x = (args.forward_speed, args.forward_speed)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    env = load_env_cls(args.task)(cfg, device="cuda:0")
    vec = load_vecenv_cls(args.task)(env, policy_group="policy", critic_group="critic")
    runner_cls = load_runner_cls(args.task) or OnPolicyRunner
    runner = runner_cls(vec, load_instinct_rl_cfg(args.task).to_dict(), device="cuda:0")
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy()
    reader = ParkourSimReader(env)
    if args.friction_coefficient is not None and not torch.allclose(
            reader.friction(), torch.full_like(reader.friction(), args.friction_coefficient), atol=1e-6):
        raise ValueError("Simulator friction differs from the requested controlled coefficient")
    obs, _ = vec.get_observations()
    offset = 0
    for name, shape in vec.get_obs_format()["policy"].items():
        size = math.prod(shape)
        if name in ("depth_image", "height_image"):
            visual_slice = slice(offset, offset + size)
            break
        offset += size
    else:
        raise ValueError("Task has no perception observation")
    n, dev = args.num_envs, env.device
    active = torch.ones(n, dtype=torch.bool, device=dev)
    steps = torch.zeros(n, device=dev)
    returns, lin_sq, yaw_sq, travel = (steps.clone() for _ in range(4))
    task_term_returns = {}
    command_nonzero, commanded_motion_stationary = steps.clone(), steps.clone()
    levels = env.scene.terrain.terrain_levels.clone()
    kinds = env.scene.terrain.terrain_types.clone()
    starts = reader.state()[0].clone()
    initial_heading = reader.state()[1][:, :2, 0].clone()
    initial_heading /= initial_heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    end_pos = starts.clone()
    result, reasons = ["step_limit"] * n, [[] for _ in range(n)]
    contact_sum = torch.zeros(reader.num_contacts, device=dev)
    flight, double_support, switches, vertical_sq, action_delta_sq = (steps.clone() for _ in range(5))
    previous_feet = reader.contact()[:, :2] > 0.5
    previous_action = reader.last_action()
    # Warp's overflow flags persist until the simulator data is reset. Observe
    # them without clearing them or changing collision buffers/physics settings.
    hfield_first_step = torch.full((n,), -1, dtype=torch.int64, device=dev)
    initial_overflow = env.sim.data.overflow.to(dtype=torch.int64)
    hfield_first_step[(initial_overflow & int(OverflowType.HFIELD)) != 0] = 0
    video = None
    trajectory = {} if args.trajectory is not None else None
    if args.video is not None:
        from instinct_mj.scripts.supermimic.parkour_video import ParkourVideo

        video = ParkourVideo(env, args.video, args.video_envs, args.video_label, levels, kinds)
    for step in range(1, math.ceil(args.seconds / env.step_dt) + 2):
        command = reader.command.command.clone()
        if args.command_mode == "forward":
            expected = command.new_tensor([args.forward_speed, 0, 0]).expand_as(command[active])
            if not torch.allclose(command[active], expected, atol=1e-6):
                raise ValueError("Actual command differs from the requested controlled forward probe")
        before = reader.state()[0]
        policy_obs = obs
        if args.perception_intervention != "intact":
            policy_obs = obs.clone()
            visual = obs[:, visual_slice]
            policy_obs[:, visual_slice] = visual.roll(1, 0) if args.perception_intervention == "shuffled" else visual.mean(0, keepdim=True)
        action = policy(policy_obs)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite evaluation action")
        obs1, reward, dones, _ = vec.step(action)
        done = dones.bool()
        hfield_hit = (env.sim.data.overflow.to(dtype=torch.int64) & int(OverflowType.HFIELD)) != 0
        hfield_first_step[active & hfield_hit & (hfield_first_step < 0)] = step
        p, R, v, w, q, qd = reader.state()
        if not all(torch.isfinite(x[active]).all() for x in (p, R, v, w, reward)):
            raise FloatingPointError("Nonfinite evaluation state or reward")
        vb = (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
        wb = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        steps += active
        returns += reward.sum(-1) * active
        # Native weighted manager values are before dt scaling. Read the cached
        # values: recomputing reward functions can mutate stateful terms.
        reconstructed_reward = torch.zeros_like(returns)
        scale = env.step_dt if env.reward_manager.scale_rewards_by_dt else 1.0
        for group, terms in env.reward_manager._termwise_reward_buf.items():
            for name, value in terms.items():
                key = f"{group}/{name}"
                task_term_returns.setdefault(key, torch.zeros_like(returns))
                task_term_returns[key] += value * scale * active
                reconstructed_reward += value * scale
        if not torch.allclose(reconstructed_reward[active], reward.sum(-1)[active], atol=1e-5, rtol=1e-5):
            raise ValueError("Reward decomposition does not match native additive task reward")
        moving_command = command[:, :2].norm(dim=-1) > 0.15
        command_nonzero += moving_command * active
        commanded_motion_stationary += moving_command * (vb[:, :2].norm(dim=-1) < 0.1) * active
        lin_sq += (vb[:, :2] - command[:, :2]).square().sum(-1) * active
        yaw_sq += (wb[:, 2] - command[:, 2]).square() * active
        travel += (p[:, :2] - before[:, :2]).norm(dim=-1) * active
        end_pos[active] = p[active]
        contact = reader.contact()
        if trajectory is not None:
            # Snapshot before any reset. Later episodes remain in arrays but are
            # excluded by valid; the first terminal transition is valid.
            sample = {"valid": active, "done": done, "terminated": env.reset_terminated,
                      "position_world": p, "rotation_body_to_world": R,
                      "linear_velocity_world": v, "angular_velocity_world": w,
                      "joint_position": q, "joint_velocity": qd, "action": action,
                      "velocity_command_body": command, "contact_indicator": contact,
                      "feet_force_history_world": reader.feet_sensor.data.force_history,
                      "feet_timers": reader.feet_timers(), "hfield_overflow": hfield_hit}
            for key, value in sample.items():
                trajectory.setdefault(key, []).append(value.detach().cpu().clone().numpy())
        feet = contact[:, :2] > 0.5
        contact_sum += (contact * active[:, None]).sum(0)
        flight += (~feet.any(-1)) * active
        double_support += feet.all(-1) * active
        switches += (feet != previous_feet).float().sum(-1) * active
        vertical_sq += v[:, 2].square() * active
        action_delta_sq += (action - previous_action).square().mean(-1) * active
        previous_feet, previous_action = feet, action.clone()
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
                "travel_xy_m": float(travel[i]), "displacement_xy_m": float((end_pos[i, :2] - starts[i, :2]).norm()),
                "progress_along_initial_heading_m": float(((end_pos[i, :2] - starts[i, :2]) * initial_heading[i]).sum()),
                "flight_fraction": float(flight[i] / steps[i].clamp_min(1)),
                "double_support_fraction": float(double_support[i] / steps[i].clamp_min(1)),
                "foot_contact_switches_per_second": float(switches[i] / (steps[i].clamp_min(1) * env.step_dt)),
                "vertical_base_velocity_rms_mps": float((vertical_sq[i] / steps[i].clamp_min(1)).sqrt()),
                "action_delta_rms": float((action_delta_sq[i] / steps[i].clamp_min(1)).sqrt())}
               for i in range(n)]
    for i, record in enumerate(records):
        record["task_reward_terms_return"] = {k: float(v[i]) for k, v in task_term_returns.items()}
        record["moving_command_fraction"] = float(command_nonzero[i] / steps[i].clamp_min(1))
        record["stationary_given_moving_command"] = float(commanded_motion_stationary[i] / command_nonzero[i]) if command_nonzero[i] else None
        record["hfield_collision_overflow_seen"] = bool(hfield_first_step[i] >= 0)
        record["hfield_collision_overflow_first_time_s"] = float(hfield_first_step[i] * env.step_dt) if hfield_first_step[i] >= 0 else None
    iteration = runner.it if hasattr(runner, "it") else runner.current_learning_iteration
    summary = {"checkpoint": str(args.checkpoint), "iteration": iteration, "seed": args.seed, "episodes": n,
               "task": args.task, "perception_intervention": args.perception_intervention,
               "terrain_probe": args.terrain_probe,
               "terrain_size_m": list(cfg.scene.terrain.terrain_generator.size),
               "command_mode": args.command_mode, "forward_speed_mps": args.forward_speed if args.command_mode == "forward" else None,
               "initial_terrain_level_max": cfg.scene.terrain.max_init_terrain_level,
               "feet_contact_threshold_N": reader.feet_threshold,
               "controlled_friction_coefficient": args.friction_coefficient,
               "episodes_with_hfield_collision_overflow": int((hfield_first_step >= 0).sum()),
               "protocol": "First episode; training randomization/terminations; pre-reset metrics; no curriculum updates",
               "survived_horizon": result.count("survived_horizon"), "failed": result.count("failure"),
               "censored": sum(x not in ("failure", "survived_horizon") for x in result),
               "mean_seconds": float(steps.mean() * env.step_dt),
               "linear_velocity_rmse_mps": float((lin_sq.sum() / steps.sum()).sqrt()),
               "yaw_velocity_rmse_radps": float((yaw_sq.sum() / steps.sum()).sqrt()),
               "contact_fraction_feet_torso_undesired": (contact_sum / steps.sum()).tolist()}
    summary["task_reward_terms_per_step"] = {k: float(v.sum() / steps.sum()) for k, v in task_term_returns.items()}
    summary["task_reward_terms_note"] = "Weighted and dt-scaled native task terms; excludes externally added AMP style reward. Sum checked against native reward every step."
    summary["moving_command_fraction"] = float(command_nonzero.sum() / steps.sum())
    summary["stationary_given_moving_command"] = float(commanded_motion_stationary.sum() / command_nonzero.sum()) if command_nonzero.sum() else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if trajectory is not None:
        import io
        import numpy as np

        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.stack(values) for key, values in trajectory.items()}
        arrays["time_s"] = np.arange(1, len(trajectory["valid"]) + 1) * env.step_dt
        arrays["joint_names"] = np.asarray(reader.robot.joint_names)
        arrays["metadata_json"] = np.asarray(json.dumps({
            "checkpoint": str(args.checkpoint), "seed": args.seed, "dt": env.step_dt,
            "alignment": "Post-step, pre-reset; valid includes first failure and excludes all subsequent episodes",
            "feet_order": ["left", "right"], "feet_timer_order": ["air_left", "air_right", "contact_left", "contact_right"],
            "force_note": "Native sensor force history, not instantaneous force; contact_indicator uses history maximum",
            "position_note": "Root position is world frame; foot positions are not exported",
            "terrain_probe": args.terrain_probe, "command_mode": args.command_mode,
        }))
        # Object-backed workspace mounts may reject ZIP header seeks. Build the
        # archive in memory, then perform one sequential write on that mount.
        archive = io.BytesIO()
        np.savez_compressed(archive, **arrays)
        # Exclusive creation also prevents accidental overwrite between checks.
        with args.trajectory.open("xb") as stream:
            stream.write(archive.getbuffer())
        summary["trajectory"] = str(args.trajectory)
    args.output.write_text(json.dumps({**summary, "records": records}, indent=2))
    print(json.dumps(summary), flush=True)
    if video is not None:
        video.close()
    env.close()


if __name__ == "__main__":
    main()
