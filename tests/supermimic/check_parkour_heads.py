"""Correctness gate of the parkour SuperMimic analytic heads (``rl/supermimic/parkour/mdp.py``): compares them against the live
env managers and FAILS (exit 1) when a tolerance is violated or a comparison is empty.

Runs the real ``Instinct-Parkour-Target-Amp-G1-v0`` env (``auto_reset=False``, finished envs are reset explicitly after their
pre-reset data was read) with zero / small random / large random actions and checks: forward kinematics of the tracked bodies,
the velocity command (pre- and post-step), the proprio frame and the flat policy observation (noise-free policy group), the
AMP window, the observation noise ranges, every reward term (incl. ``volume_points_penetration``), the feet timers, hard
terminations (natural and forced on both sides of every threshold) and the shape / finiteness / gradients of the rest.

Conventions and findings (also see ``ParkourMDP.reward``):

- The env's state ``S0`` / ``S1`` are read from ``qpos`` / ``qvel`` before / after ``env.step``. The command PARAMETERS ``cmd0``
  are read before the step (the env resamples them after computing the reward), ``cmd1`` after.
- TIMING: mjlab evaluates the reward before the command manager update, so rewards use the command of ``S0``
  (``vel_command_b`` read before the step); the table lists the S0 / S1 candidates of the command-dependent terms
  ("[S1 variant]" rows) to document that choice. The ``feet_at_plane`` rays are cast by ``sim.sense()`` at the end of the
  previous step, i.e. from the feet of ``S0``.
- STALENESS: the env evaluates terminations and rewards with derived quantities (``xpos``, ``cvel``, ...) that lag ``qpos``
  by one physics substep (0.005 s). The analytic heads use the exact post-step state, so terms that depend on body poses /
  velocities (feet, pelvis, volume points) and on torques differ by O(v * 0.005) / O(a * 0.005).

Tolerances (max abs err, mean abs err) are therefore of two kinds: "exact" terms (functions of ``qpos`` / ``qvel`` / actions)
use 1e-4 .. 1e-5 and the rest are set from the measured substep-staleness error with a margin (comment next to each):

- ``dof_acc_l2`` is APPROXIMATED as ((qd1 - qd0) / dt)^2 (env: ``qacc`` of the last substep): large relative error, only a
  sanity bound on the mean is checked.
- ``dof_torques_l2`` / ``energy`` / ``torque_limits`` use the analytic PD torque of the post-step state and the NEW action,
  the env reads ``qfrc_actuator`` of the last substep (possibly with a lagged action from the actuator delay): bounded errors.
- Feet timers: ``advance_feet_timers`` is a per-env-step approximation of the per-substep sensor update; its error is reported
  separately ("timers: advance_feet_timers vs env") and rewards are checked with the REAL timers and REAL contact indicators.
- The contact indicator ("max force over history > 1 N") and the env's ``found`` based timers can disagree, which makes
  ``feet_air_time`` differ in a few envs (reported as the max error).

    .venv/bin/python tests/supermimic/check_parkour_heads.py
"""

from __future__ import annotations

import argparse
import sys

import torch

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl.supermimic.parkour.mdp import (
    AMP_HIST,
    AMP_TERMS,
    HIST,
    PROPRIO_DIM,
    PROPRIO_TERMS,
    XI_DIM,
    build_parkour_mdp,
)
from instinct_mj.rl.supermimic.parkour.sim import ParkourSimReader
from instinct_mj.rl.supermimic.rotations import mv as mv_
from instinct_mj.rl.supermimic.rotations import quat_to_mat, yaw_mat
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls

TASK = "Instinct-Parkour-Target-Amp-G1-v0"

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

dev = "cuda:0"
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.auto_reset = False  # as in the supermimic runner: terminal states stay readable
cfg.observations["policy"].enable_corruption = False  # noise-free policy group, comparable with the analytic assembly
env = load_env_cls(TASK)(cfg=cfg, device=dev, render_mode=None)
rd = ParkourSimReader(env)
robot = env.scene["robot"]
kin, scanner, mdp = build_parkour_mdp(env, 0.05, "~/.cache/instinct_mj/supermimic_heightmaps", dev)
N = args.num_envs
dt = env.step_dt
track_ids = robot.find_bodies([kin.body_names[i] for i in kin.tracked_idx], preserve_order=True)[0]  # all bodies, TRACKED first

stats: dict[str, list[torch.Tensor]] = {}
tol: dict[str, tuple[float, float | None]] = {}
failures: list[str] = []


def log(name: str, err: torch.Tensor, max_tol: float, mean_tol: float | None = None) -> None:
    tol[name] = (max_tol, mean_tol)
    stats.setdefault(name, []).append(err.detach().abs().flatten().float().cpu())


def fail_if(cond: bool, msg: str) -> None:
    if cond:
        failures.append(msg)


def grp(obs: dict, name: str, order) -> torch.Tensor:
    return torch.cat([obs[name][t].flatten(1) for t in order], -1)


# ---- static checks ----
pol_terms = env.cfg.observations["policy"].terms
noise = mdp.proprio_noise
for name, (a, b) in PROPRIO_TERMS.items():
    t = pol_terms[name]
    expect = 0.0 if t.noise is None else float(t.noise.n_max) * (1.0 if t.scale is None else float(t.scale))
    log(f"noise half width (scaled): {name}", noise[a:b] - expect, 0.0, 0.0)
fail_if(noise.shape != (PROPRIO_DIM,), "proprio_noise shape")
z = mdp.add_proprio_noise(torch.zeros(4096, PROPRIO_DIM, device=dev))
log("noise samples exceed half width", torch.relu(z.abs() - noise), 0.0, 0.0)
fail_if(not bool((z.abs().amax(0)[noise > 0] > 0.9 * noise[noise > 0]).all()), "noise samples do not reach the half width")

# ---- stepping ----
env.reset()
rng = torch.Generator(device=dev).manual_seed(args.seed)
n_act = env.action_manager.total_action_dim
proprio_hist: list[torch.Tensor] = []
amp_hist: list[torch.Tensor] = []
S_hist: list = []
a_hist: list = []
cmd_hist: list = []
since_reset = torch.zeros(N, dtype=torch.long, device=dev)
n_done = 0
n_vp_active = 0
tm = env.termination_manager
for step in range(args.steps):
    phase = step * 3 // args.steps
    if phase == 0:
        action = torch.zeros(N, n_act, device=dev)
    else:
        action = (0.3 if phase == 1 else 1.0) * torch.randn(N, n_act, device=dev, generator=rng)
    S0, a_prev, cmd0 = rd.state(), rd.last_action(), rd.command_params()
    timers0 = rd.feet_timers()
    vcmd0_env = rd.command.vel_command_b.clone()
    obs, rew, terminated, truncated, extras = env.step(action)
    rew_total = rew["rewards"].clone()
    # the manager keeps weight * term (without the dt scaling)
    env_terms = {k: v.clone() * dt for k, v in env.reward_manager._termwise_reward_buf["rewards"].items()}
    a = env.action_manager.action.clone()
    done = terminated | truncated
    since_reset += 1
    S1, cmd1, contact1, timers1 = rd.state(), rd.command_params(), rd.contact(), rd.feet_timers()
    vcmd1_env = rd.command.vel_command_b.clone()
    d = robot.data

    # ---- FK ----
    bp, bR, bv, bw = kin.forward(*S1)
    log("fk: tracked pos", bp - d.body_link_pos_w[:, track_ids], 1e-4, 1e-5)
    log("fk: tracked rot", bR - quat_to_mat(d.body_link_quat_w[:, track_ids]), 1e-4, 1e-5)
    log("fk: tracked lin vel", bv - d.body_link_lin_vel_w[:, track_ids], 1e-3, 1e-4)
    log("fk: tracked ang vel", bw - d.body_link_ang_vel_w[:, track_ids], 1e-3, 1e-4)

    # ---- velocity command (the env buffer before / after the step) ----
    log("velocity_command(S0, cmd0) vs env command before step", mdp.velocity_command(S0, cmd0) - vcmd0_env, 1e-4, 1e-5)
    log("velocity_command(S1, cmd1) vs env command after step", mdp.velocity_command(S1, cmd1) - vcmd1_env, 1e-4, 1e-5)

    # ---- timers / contacts ----
    # report-only magnitude: the force-history indicator disagrees with the env's ``found`` based timers in ~5% of the feet-steps
    log("timers: advance_feet_timers(real indicator) vs env timers", mdp.advance_feet_timers(timers0, contact1) - timers1, 1e9, 2e-2)
    ct_found = (timers1[:, 2:4] > 0).float()
    both = (contact1[:, :2] == ct_found).all(-1, keepdim=True).float().expand(-1, 4)
    log("timers: advance_feet_timers vs env where indicator == found (per-substep vs per-step update)", (mdp.advance_feet_timers(timers0, contact1) - timers1) * both, 1e9, 1e-2)  # max: a foot that touches the ground for one substep resets the env timers (heavy tail)
    ct_ind = (timers1[:, 2:4] > 0).float()
    log("contact: force indicator vs (env contact timer > 0)", contact1[:, :2] - ct_ind, 1.0, 0.1)
    log("contact: indicator=1 but env timer says air", (contact1[:, :2] * (1 - ct_ind)), 1.0, 0.1)
    log("contact: indicator=0 but env timer says contact", ((1 - contact1[:, :2]) * ct_ind), 1.0, 0.1)

    # ---- applied torque: analytic PD (post-step state, new action) vs qfrc_actuator after the step's final ``forward()``
    # (fresh at S1; the reward manager sees the value of the last substep, one substep stale -> torque_limits / energy rows) ----
    qfrc = env.sim.data.qfrc_actuator[:, rd.joint_v]
    log("torque: analytic PD(S1, a) vs qfrc_actuator (all joints)", mdp._torque(S1[4], S1[5], a) - qfrc, 1e9, 3.0)
    log("torque: analytic PD(S0, a_prev) vs qfrc_actuator (all joints)", mdp._torque(S0[4], S0[5], a_prev) - qfrc, 1e9, 1e9)

    # ---- rewards ----
    S1g = tuple(x.clone().requires_grad_(True) for x in S1)
    r, terms = mdp.reward(S0, S1g, action, a_prev, cmd0, timers1, contact1, terms=True)
    # the env's ``a`` is the processed action: action_rate etc. use ``env.action_manager.action`` == ``action``
    alive = (~terminated).float()
    # (max tol, mean tol) per term; see the header. Exact terms: functions of qpos / qvel / action.
    T = {
        "track_lin_vel_xy_exp": (4e-2, 1.5e-3),  # v of S1 vs substep-stale root velocity (see header); max: full range of the term
        "track_ang_vel_z_exp": (4e-2, 5e-3),
        "heading_error": (1e-4, 1e-5),
        "dont_wait": (1.0, 5e-3),  # step function of the stale velocity
        "is_alive": (1e-6, 1e-7),
        "stand_still": (1e-4, 1e-5),
        "feet_air_time": (2e-3, 1e-5),  # timer / indicator disagreement in some envs
        "feet_slide": (1.5e-2, 1e-3),  # stale foot velocity (and indicator vs history differences)
        "joint_deviation_hip": (1e-5, 1e-6),
        "ang_vel_xy_l2": (5e-2, 7e-4),  # stale body angular velocity
        "dof_torques_l2": (1e-4, 1e-5),  # lagged action / substep state: small because the weight is tiny
        "dof_acc_l2": (1e9, 1e9),  # approximation, only the mean sanity bound below
        "dof_vel_l2": (1e-4, 1e-5),
        "action_rate_l2": (1e-6, 1e-7),
        "flat_orientation_l2": (5e-3, 4e-4),  # orientation of the substep-stale root pose
        "pelvis_orientation_l2": (5e-3, 4e-4),
        "feet_flat_ori": (2e-3, 2e-4),
        "feet_at_plane": (3e-3, 3e-4),
        "feet_close_xy": (4e-3, 2e-4),
        "energy": (1e-3, 1e-4),
        "freeze_upper_body": (1e-6, 1e-7),
        "dof_pos_limits": (1e-6, 1e-7),
        "dof_vel_limits": (1e-5, 1e-6),
        "torque_limits": (0.2, 2e-4),  # squared excess of the lagged / saturated torque, see the torque rows
        "undesired_contacts": (1e-6, 1e-7),
    }
    for k, v in terms.items():
        ref = env_terms[k]
        v = v * alive if k == "is_alive" else v
        if k == "dof_acc_l2":
            log(f"reward[{k}] (approx) vs env, in units of dt*|weight|", (v - ref) / (abs(mdp.w[k]) * dt), 1e12, None)
            log(f"reward[{k}] (approx) mean sanity: |mine| / |env| - 1 on the batch mean", ((v.abs().mean() / ref.abs().mean()) - 1).reshape(1), 1e9, 1.0)
            continue
        log(f"reward[{k}] vs env", v - ref, *T[k])
    if phase == 0:
        for k in ("track_ang_vel_z_exp", "ang_vel_xy_l2", "feet_slide", "flat_orientation_l2", "torque_limits"):
            log(f"reward[{k}] vs env [zero-action phase only]", terms[k] - env_terms[k], 1e9, None)
    fail_if(set(terms) != set(env_terms) - {"volume_points_penetration"}, f"term sets differ: {set(env_terms) ^ set(terms)}")
    # timing variants: command / rays from S1 instead of S0 (documents the choice, report only; loose bounds)
    _, t_s1 = mdp.reward(S1, S1, action, a_prev, cmd1, timers1, contact1, terms=True)
    for k in ("heading_error", "track_ang_vel_z_exp", "feet_at_plane"):
        log(f"reward[{k}] [S1 variant] vs env", t_s1[k] - env_terms[k], 1e9, None)
    # soft timers (the world-model path): reward with the advanced timers / contact; report the extra error
    r_adv = mdp.reward(S0, S1, action, a_prev, cmd0, mdp.advance_feet_timers(timers0, contact1), contact1)
    log("reward total: advanced timers vs real timers", r_adv - r, 1e9, None)

    vp = mdp.volume_points_penetration(S1g)
    vp_env = env_terms["volume_points_penetration"]
    n_vp_active += int((vp_env != 0).sum())
    log("reward[volume_points_penetration] vs env", vp - vp_env, 5e-2 * abs(mdp.w["volume_points_penetration"]) * dt, 5e-3 * abs(mdp.w["volume_points_penetration"]) * dt)
    total = r + vp
    log("reward total (+ vp, is_alive masked) vs env", total - (1 - alive) * mdp.w["is_alive"] * dt - rew_total, 0.3, 8e-3)  # dominated by dof_acc_l2 and the stale terms above
    if step % 20 == 0:
        (total.sum()).backward()
        fail_if(not all(bool(torch.isfinite(x.grad).all()) for x in S1g), "non-finite reward gradient")

    # ---- terminations ----
    flag_env = tm.get_term("base_contact") | tm.get_term("bad_orientation") | tm.get_term("root_height")
    hard = mdp.hard_termination(S1, cmd1, contact1)
    log("termination natural: flag mismatch", (hard != flag_env).float(), 1.0, 0.01)
    soft = mdp.soft_termination(S1, cmd1, contact1)
    fail_if(bool(((soft < -1e-6) | (soft > 1 + 1e-6)).any()) or not bool(torch.isfinite(soft).all()), "soft termination out of [0,1]")
    log("soft termination: mean prob where hard=1 (1 - p)", (1 - soft)[hard], 1.0, 0.9)

    # ---- observations: proprio frame, flat policy obs, amp window ----
    frame = mdp.proprio(S1, a, cmd1)
    S_hist.append(S1), a_hist.append(a), cmd_hist.append(cmd1)
    proprio_hist.append(frame)
    amp_hist.append(mdp.amp_frame(S1))
    proprio_hist, amp_hist = proprio_hist[-HIST:], amp_hist[-AMP_HIST:]
    S_hist, a_hist, cmd_hist = S_hist[-AMP_HIST:], a_hist[-AMP_HIST:], cmd_hist[-AMP_HIST:]
    pol = obs["policy"]
    last = {n: pol[n].reshape(N, HIST, -1)[:, -1] for n in PROPRIO_TERMS}
    for n, (lo, hi) in PROPRIO_TERMS.items():
        log(f"proprio frame: {n} vs env newest frame", frame[:, lo:hi] - last[n], 1e-4, 1e-5)
    ok = (since_reset >= max(HIST, AMP_HIST)) & ~done
    if len(proprio_hist) == HIST and len(amp_hist) == AMP_HIST and bool(ok.any()):
        order = env.observation_manager.active_terms["policy"]
        fail_if(order != [*PROPRIO_TERMS, "depth_image"], f"unexpected policy order {order}")
        depth = pol["depth_image"]
        mine = mdp.policy_obs(torch.stack(proprio_hist, 1), depth)
        env_flat = torch.cat([pol[n].flatten(1) for n in order], -1)
        n_flat = env_flat.shape[-1] - depth[0].numel()
        log("policy obs flat (all but depth) vs env", (mine - env_flat)[ok][:, :n_flat], 1e-4, 1e-5)
        log("policy obs flat: depth part", (mine - env_flat)[ok][:, n_flat:], 0.0, 0.0)
        amp_env = grp(obs, "amp_policy", AMP_TERMS)
        amp_mine = mdp.amp_window(torch.stack(amp_hist, 1))
        log("amp window vs env amp_policy", (amp_mine - amp_env)[ok], 1e-4, 1e-5)
        # leading-dim variants (runner usage): stacked states, (B, T, ...)
        Sh = tuple(torch.stack([s[i] for s in S_hist], 1) for i in range(6))
        ah, ch = torch.stack(a_hist, 1), torch.stack(cmd_hist, 1)
        log("leading dims: proprio(S_hist) vs loop", mdp.proprio(Sh, ah, ch)[:, -HIST:] - torch.stack(proprio_hist, 1), 1e-5, 1e-6)
        log("leading dims: amp_frame(S_hist) vs loop", mdp.amp_frame(Sh)[:, -AMP_HIST:] - torch.stack(amp_hist, 1), 1e-5, 1e-6)
        xi_n = torch.zeros(N, XI_DIM, device=dev)
        cf = mdp.critic_features(S1, cmd1, timers1, xi_n)
        fail_if(cf.shape != (N, mdp.critic_dim) or not bool(torch.isfinite(cf).all()), f"critic_features {tuple(cf.shape)} vs {mdp.critic_dim}")
        cfh = mdp.critic_features(Sh, ch, timers1.unsqueeze(1).expand(-1, ch.shape[1], -1), xi_n.unsqueeze(1).expand(-1, ch.shape[1], -1))
        log("leading dims: critic_features(S_hist)[-1] vs critic_features(S1)", cfh[:, -1] - cf, 1e-5, 1e-6)
        sc = mdp.scan(S1)
        log("scan vs height_scan", sc - scanner.height_scan(S1[0], S1[1]), 0.0, 0.0)

    # ---- feet ground height: heightmap lookup vs the env's ray hits (rays cast from S1 feet at the end of the step) ----
    if step % 5 == 0:
        for side, sname in enumerate(("left_height_scanner", "right_height_scanner")):
            hit = env.scene[sname].data
            miss = hit.distances < 0
            Rf, pf = bR[:, 1 + side], bp[:, 1 + side]
            xy = pf[:, None, :2] + (yaw_mat(Rf)[:, None, :2, :2] * mdp.foot_rays[None, :, None, :]).sum(-1)
            gz = scanner.ground_height(xy)
            log(f"feet ground height: heightmap vs env hit z ({sname}, hits only)", torch.where(miss, torch.zeros_like(gz), gz - hit.hit_pos_w[..., 2]), 1.0, 5e-3)  # max: heightmap resolution at edges (0.05 m cells) / overhangs

    # ---- explicit reset of finished envs ----
    n_done += int(done.sum())
    if done.any():
        ids = done.nonzero().squeeze(-1)
        env.reset(env_ids=ids)
        since_reset[ids] = 0
        proprio_hist, amp_hist, S_hist, a_hist, cmd_hist = [], [], [], [], []

# ---- volume points: teleport the left foot into the virtual edge cylinders so that the env penetration is non-zero ----
vp_sensor = env.scene["leg_volume_points"]
cyl = torch.as_tensor(vp_sensor._virtual_obstacles["edges"].cylinders.cylinders_np, device=dev)
fail_if(cyl.shape[0] == 0, "no virtual edge cylinders")
w_vp = abs(mdp.w["volume_points_penetration"]) * dt
n_vp_exact = 0
for rnd in range(12):
    env.reset()
    env.step(torch.zeros(N, n_act, device=dev))
    S0 = rd.state()
    bp_, bR_ = kin.forward(*S0)[:2]
    idx = torch.randint(0, cyl.shape[0], (N,), device=dev, generator=rng)
    mid = 0.5 * (cyl[idx, :3] + cyl[idx, 3:6])
    target = mid - mv_(bR_[:, 1], torch.tensor([0.05, 0.0, -0.043], device=dev))
    pose = torch.cat((S0[0] + target - bp_[:, 1], torch.nn.functional.normalize(robot.data.root_link_quat_w.clone(), dim=-1)), -1)
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    S0 = rd.state()
    # exact (non-stale) check of the formula: env sensor refreshed from the forwarded teleported state vs the analytic head
    vp_sensor._invalidate_cache()
    from instinct_mj.tasks.parkour.mdp.rewards import volume_points_penetration as vp_fn

    exact = vp_fn(env, "leg_volume_points") * mdp.w["volume_points_penetration"] * dt
    log("reward[volume_points_penetration] vs env sensor on a freshly forwarded state (no staleness), units of dt*|weight|", (mdp.volume_points_penetration(S0) - exact) / w_vp, 5e-2, 5e-3)
    n_vp_exact += int((exact != 0).sum())
    obs, rew, terminated, truncated, extras = env.step(torch.zeros(N, n_act, device=dev))
    S1 = rd.state()
    vp_env = env.reward_manager._termwise_reward_buf["rewards"]["volume_points_penetration"].clone() * dt
    S1g = tuple(x.clone().requires_grad_(True) for x in S1)
    vp = mdp.volume_points_penetration(S1g)
    n_vp_active += int((vp_env != 0).sum())
    log("reward[volume_points_penetration] vs env [teleported into edges], in units of dt*|weight|", (vp - vp_env) / w_vp, 1e9, 12.0)
    active = vp_env != 0
    if bool(active.any()):
        log("reward[volume_points_penetration] relative error on active envs |mine - env| / |env|", ((vp - vp_env).abs() / vp_env.abs().clamp(min=1e-9))[active], 1e9, 0.9)  # staleness + violent teleport dynamics, see the header
        # gradient: first-order change along the gradient direction of q (finite difference of the exact-offset value)
        vp.sum().backward()
        g = S1g[4].grad
        gn = g.norm(dim=-1, keepdim=True)
        eps = 1e-4
        S1p = (*S1[:4], S1[4] + eps * g / gn.clamp(min=1e-12), S1[5])
        pred = eps * gn.squeeze(-1)
        actual = (mdp.volume_points_penetration(S1p) - vp).detach()
        sel = active & (gn.squeeze(-1) > 1e-6)
        fail_if(not bool(torch.isfinite(g).all()), "non-finite volume point gradient")
        if bool(sel.any()):
            log("volume points gradient: |actual change - first-order prediction| / |prediction| (eps 1e-4 rad)", ((actual - pred).abs() / pred.abs())[sel], 1e9, 0.5)

# ---- forced termination checks: write perturbed states so both sides of every threshold are hit ----
env.reset()
env.step(torch.zeros(N, n_act, device=dev))
forced_true: dict[str, list[torch.Tensor]] = {}
for case, offsets in (("height", (0.3, -0.3)), ("tilt", (0.5, 1.5))):
    for sign_i in range(2):
        S0 = rd.state()
        pose = torch.cat((S0[0], torch.nn.functional.normalize(robot.data.root_link_quat_w.clone(), dim=-1)), -1)
        origin_z = rd.command_params()[:, 13].clamp(max=0.0)
        if case == "height":
            pose[:, 2] = origin_z + mdp.min_height + offsets[sign_i]
        else:
            half = torch.full((N,), offsets[sign_i] / 2, device=dev)
            qt = torch.stack((half.cos(), half.sin(), torch.zeros_like(half), torch.zeros_like(half)), -1)
            w1, x1, y1, z1 = qt.unbind(-1)
            w2, x2, y2, z2 = pose[:, 3:7].unbind(-1)
            pose[:, 3:7] = torch.stack((w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2), -1)  # fmt: skip
        robot.write_root_link_pose_to_sim(pose)
        env.sim.forward()
        tm.compute()
        Sf, cmdf, contactf = rd.state(), rd.command_params(), rd.contact()
        hard = mdp.hard_termination(Sf, cmdf, contactf)
        for k in ("bad_orientation", "root_height"):
            flag = tm.get_term(k)
            forced_true.setdefault(k, []).append(flag.float().cpu())
        flag = tm.get_term("base_contact") | tm.get_term("bad_orientation") | tm.get_term("root_height")
        log(f"termination forced ({case} {offsets[sign_i]:+.1f}): flag mismatch", (hard != flag).float(), 0.0, 0.0)
        env.reset(env_ids=torch.arange(N, device=dev))
        env.step(torch.zeros(N, n_act, device=dev))

# ---- report + gates ----
print(f"{'quantity':100s} {'max abs err':>12s} {'mean abs err':>13s}  {'tol max':>9s} {'tol mean':>9s}  n")
for name, errs in stats.items():
    e = torch.cat(errs)
    mx, mean = e.max().item(), e.mean().item()
    mt, at = tol[name]
    print(f"{name:100s} {mx:12.3e} {mean:13.3e}  {mt:9.1e} {('-' if at is None else f'{at:.1e}'):>9s}  {len(e)}")
    fail_if(len(e) == 0, f"empty comparison: {name}")
    fail_if(mx > mt, f"{name}: max err {mx:.3e} > {mt:.3e}")
    if at is not None:
        fail_if(mean > at, f"{name}: mean err {mean:.3e} > {at:.3e}")
for k, v in forced_true.items():
    e = torch.cat(v)
    n_true, n_false = int(e.sum()), int((1 - e).sum())
    print(f"forced {k}: env flag true {n_true}, false {n_false}")
    fail_if(n_true == 0 or n_false == 0, f"forced termination check for {k} did not hit both sides")
print(f"terminal (done) transitions seen: {n_done}; steps with volume-point penetration in the env: {n_vp_active}")
fail_if(n_done == 0, "no terminal transition was exercised")
fail_if(n_vp_active == 0 or n_vp_exact == 0, "no volume point penetration was exercised")
if failures:
    print("FAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
