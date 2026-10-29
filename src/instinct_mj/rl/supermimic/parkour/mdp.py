"""Analytic (differentiable) task heads of the parkour task (``Instinct-Parkour-Target-Amp-G1-v0``) for SuperMimic.

Everything the imagination needs that is not learned: the velocity command derived from the target, the policy / AMP
observations in EXACTLY the env's format, the reward terms of the ``rewards`` group, the terminations and the critic
features, all as torch functions of the floating-base state. All constants (weights, thresholds, gains, limits, command
settings) are read once from the live env in ``__init__``.

Conventions (see ``sim.py`` for the layouts of the command parameters ``cmd``, the contacts and the feet timers):

- ``S = (p, R, v, w, q, qd)``: root (``pelvis``) position (B, 3), rotation matrix (B, 3, 3), world linear velocity of the
  root-frame origin (B, 3), world angular velocity (B, 3), joint positions (B, 29) and velocities (B, 29), joint order of
  the robot entity.
- ``cmd``: (B, CMD_DIM) command parameters. ``contact``: (B, NUM_CONTACTS) indicators or probabilities.
- Rewards are the env's per-step rewards: weight * term * dt (``scale_rewards_by_dt``).
- All functions are traced by ``torch.compile``: no host syncs, no data-dependent shapes (except
  ``volume_points_penetration``, which runs the env's warp kernel and is excluded from compilation).
"""

from __future__ import annotations

import inspect

import mujoco
import torch
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.utils.lab_api.string import resolve_matching_names_values

from instinct_mj.utils.smooth_rewards import smooth_dont_wait_penalty

from ..kinematics import G1Kinematics
from ..mdp import INV_DIM, invariant_features
from ..rotations import mv, yaw_mat
from ..sim import HEIGHT_SCANNER_MAX_DISTANCE, HEIGHT_SCANNER_RESOLUTION, HEIGHT_SCANNER_SIZE
from ..terrain import HeightScanner, load_or_build_heightmap
from .sim import CMD_MAX, CMD_ORIGIN, CMD_RANDOM, CMD_RANDOM_VEL, CMD_STANDING, CMD_TARGET

NUM_J = 29
HIST = 8  # history length of the proprioceptive policy terms
AMP_HIST = 10  # history length of the amp_policy terms
PROPRIO_DIM = 96  # one frame: base_ang_vel 3, projected_gravity 3, velocity_commands 3, joint_pos 29, joint_vel 29, actions 29
AMP_FRAME_DIM = 67  # one frame: projected_gravity 3, joint_pos_rel 29, joint_vel 29, base_lin_vel 3, base_ang_vel 3
DEPTH_SHAPE = (8, 18, 32)  # policy depth_image term: 8 frames of the processed depth camera
XI_DIM = NUM_J + 1  # normalized context: actuation delay of every joint (``sim.actuator_lags``), friction (``sim.friction``)

TRACKED = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
K_PELVIS, K_LEFT, K_RIGHT = 0, 1, 2
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
# one frame of the policy group: (start, end) of every term, in the env's term order
PROPRIO_TERMS = {"base_ang_vel": (0, 3), "projected_gravity": (3, 6), "velocity_commands": (6, 9), "joint_pos": (9, 38),
                 "joint_vel": (38, 67), "actions": (67, 96)}  # fmt: skip
AMP_TERMS = {"projected_gravity": (0, 3), "joint_pos_rel": (3, 32), "joint_vel": (32, 61), "base_lin_vel": (61, 64),
             "base_ang_vel": (64, 67)}  # fmt: skip


def _norm(x: torch.Tensor) -> torch.Tensor:
    """Euclidean norm over the last dim with a finite gradient at 0 (bias <= 1e-6)."""
    return torch.sqrt((x * x).sum(-1) + 1e-12)


class ParkourMDP:
    def __init__(self, env, kin: G1Kinematics, scanner: HeightScanner, device: str):
        """``kin`` tracks (at least) ``pelvis``, ``left_ankle_roll_link`` and ``right_ankle_roll_link``; ``scanner`` is the
        terrain heightmap (``scan`` / ``ground_height``). Reads every constant from ``env`` (the unwrapped env)."""
        self.kin, self.scanner, self.device = kin, scanner, device
        self.dt = env.step_dt
        robot = env.scene["robot"]
        rm, tm = env.reward_manager, env.termination_manager
        # ---- command ----
        c = env.command_manager.get_term("base_velocity").cfg
        assert c.only_positive_lin_vel_x, "only the only_positive_lin_vel_x=True branch of the command is implemented"
        self.cmd_cfg = c
        # ---- nominal joint data ----
        self.default_q = robot.data.default_joint_pos[0].to(device)
        self.default_qd = robot.data.default_joint_vel[0].to(device)
        limits = robot.data.soft_joint_pos_limits[0]
        self.soft_lo, self.soft_hi = limits[:, 0].to(device), limits[:, 1].to(device)
        self.act_scale = env.action_manager.get_term("joint_pos").scale[0].to(device)
        model = env.sim.mj_model
        self.kp = torch.zeros(NUM_J, device=device)
        self.kd = torch.zeros(NUM_J, device=device)
        self.effort_limit = torch.zeros(NUM_J, device=device)
        for act in robot.actuators:
            n = len(act.target_ids)
            ctrl = [int(i) for i in act.global_ctrl_ids]
            for k, j in enumerate(act.target_ids.tolist()):
                self.kp[j] = float(model.actuator_gainprm[ctrl[k], 0])
                self.kd[j] = float(model.actuator_gainprm[ctrl[n + k], 0])
                self.effort_limit[j] = float(abs(model.jnt_actfrcrange[int(robot.indexing.joint_ids[j])]).max())
        # ---- observation scales / noise (noise is applied BEFORE the scale in the env) ----
        pol = env.cfg.observations["policy"].terms
        assert list(env.observation_manager.active_terms["policy"]) == [*PROPRIO_TERMS, "depth_image"]
        scale = torch.ones(PROPRIO_DIM, device=device)
        noise = torch.zeros(PROPRIO_DIM, device=device)
        for name, (a, b) in PROPRIO_TERMS.items():
            term = pol[name]
            s_ = 1.0 if term.scale is None else float(term.scale)
            scale[a:b] = s_
            if term.noise is not None:
                assert term.noise.n_min == -term.noise.n_max
                noise[a:b] = float(term.noise.n_max) * s_
        self.proprio_scale, self.proprio_noise = scale, noise
        amp = env.cfg.observations["amp_policy"].terms
        assert list(env.observation_manager.active_terms["amp_policy"]) == list(AMP_TERMS)
        self.amp_vel_scale = 1.0 if amp["joint_vel"].scale is None else float(amp["joint_vel"].scale)
        # ---- rewards ----
        self.w = {n: rm.get_term_cfg(n).weight for n in rm.active_terms["rewards"]}
        self.params = {n: rm.get_term_cfg(n).params for n in rm.active_terms["rewards"]}
        self._funcs = {n: rm.get_term_cfg(n).func for n in rm.active_terms["rewards"]}
        self.joint_ids = {}
        for n in rm.active_terms["rewards"]:
            ac = self.params[n].get("asset_cfg")
            if ac is not None and ac.joint_ids is not None:
                self.joint_ids[n] = torch.arange(NUM_J)[ac.joint_ids].to(device)
        for n in ("feet_slide", "feet_flat_ori", "feet_at_plane", "feet_close_xy"):
            assert tuple(robot.body_names[i] for i in self.params[n]["asset_cfg"].body_ids) == FEET
        assert tuple(robot.body_names[i] for i in self.params["pelvis_orientation_l2"]["asset_cfg"].body_ids) == ("pelvis",)
        tracked_names = [kin.body_names[i] for i in kin.tracked_idx]
        assert tracked_names[: len(TRACKED)] == TRACKED
        names = robot.find_joints(self.params["dof_vel_limits"]["asset_cfg"].joint_names)[1]
        self.vel_limits = torch.tensor(
            resolve_matching_names_values(data=self.params["dof_vel_limits"]["velocity_limits"], list_of_strings=names)[2],
            device=device,
        )
        scanner_cfg = env.scene["left_height_scanner"].cfg
        assert env.scene["right_height_scanner"].cfg.pattern.generate_rays(None, "cpu")[0].equal(
            scanner_cfg.pattern.generate_rays(None, "cpu")[0]
        )
        self.foot_rays = scanner_cfg.pattern.generate_rays(None, device)[0][:, :2]  # (2, 2) offsets in the foot yaw frame
        self.ray_max_distance = scanner_cfg.max_distance
        # ---- volume points ----
        self.vp_sensor = env.scene[self.params["volume_points_penetration"]["sensor_name"]]
        assert self.vp_sensor.body_names == list(FEET)
        self.vp_pattern = self.vp_sensor._volume_points_pattern.to(device)  # (P, 3) in the link frame
        # Centre-of-mass tables: the sensor converts ``cvel`` (referenced to the COM of the whole robot) with the subtree COM of
        # the link itself, which shifts its linear velocity by w x (com_robot - com_link_subtree); replicated in
        # ``volume_points_penetration``. Every body is attached to its nearest jointed ancestor (all of them are tracked).
        mj = model
        data = mujoco.MjData(mj)
        mujoco.mj_forward(mj, data)
        id2name = dict(zip(robot.indexing.body_ids.tolist(), robot.body_names))
        mass = torch.zeros(len(tracked_names), device=device)
        moment = torch.zeros(len(tracked_names), 3, device=device)  # sum of m * (com position in the ancestor frame)
        for bid in robot.indexing.body_ids.tolist():
            anc = bid
            while mj.body_jntnum[anc] == 0:
                anc = int(mj.body_parentid[anc])
            k = tracked_names.index(id2name[anc])
            com_w = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ mj.body_ipos[bid]
            rel = data.xmat[anc].reshape(3, 3).T @ (com_w - data.xpos[anc])
            mass[k] += float(mj.body_mass[bid])
            moment[k] += float(mj.body_mass[bid]) * torch.as_tensor(rel, device=device, dtype=torch.float32)
        self.com_mass, self.com_moment = mass, moment
        # ---- terminations ----
        self.limit_angle = tm.get_term_cfg("bad_orientation").params["limit_angle"]
        self.min_height = tm.get_term_cfg("root_height").params["minimum_height"]
        self.critic_dim = INV_DIM + 3 + 3 + 3 + 2 + 4 + XI_DIM + scanner.num_rays

    dt: float
    proprio_noise: torch.Tensor  # (PROPRIO_DIM,) half widths of the uniform noise in the SCALED units of the frame
    critic_dim: int

    def _param(self, term: str, key: str):
        """Reward parameter: the cfg value, else the default of the reward function."""
        if key in self.params[term]:
            return self.params[term][key]
        return inspect.signature(self._funcs[term]).parameters[key].default

    # ---------------- command / observations ----------------
    def velocity_command(self, S, cmd) -> torch.Tensor:
        """(..., 3) ``vel_command_b`` of ``PoseVelocityCommand._update_command`` for the state ``S``."""
        p, R = S[0], S[1]
        c = self.cmd_cfg
        target_vec = cmd[..., CMD_TARGET] - p
        target_dist = torch.linalg.vector_norm(target_vec[..., :2], dim=-1)
        pos_b = mv(yaw_mat(R).transpose(-1, -2), target_vec)  # quat_rotate_inverse(yaw_quat(root_quat), target_vec)
        vxy = pos_b[..., :2] * c.velocity_control_stiffness
        target_direction = torch.atan2(target_vec[..., 1], target_vec[..., 0])
        heading_w = torch.atan2(R[..., 1, 0], R[..., 0, 0])  # root heading: yaw of the x axis
        wz = wrap_to_pi(target_direction - heading_w) * c.heading_control_stiffness
        max_cmd = cmd[..., CMD_MAX]
        vx = torch.clamp(vxy[..., 0], min=torch.zeros_like(max_cmd[..., 0]), max=max_cmd[..., 0])  # only_positive_lin_vel_x
        vy = torch.clamp(vxy[..., 1], min=-max_cmd[..., 1], max=max_cmd[..., 1])
        wz = torch.clamp(wz, c.ranges.ang_vel_z[0], c.ranges.ang_vel_z[1])
        near = (target_dist > c.target_dis_threshold).float()
        vx, vy, wz = vx * near, vy * near, wz * near
        lin_ok = (torch.sqrt(vx * vx + vy * vy) > c.lin_vel_threshold).float()
        vx, vy = vx * lin_ok, vy * lin_ok
        wz = wz * (wz.abs() > c.ang_vel_threshold).float()
        v = torch.stack((vx, vy, wz), -1)
        v = torch.where(cmd[..., CMD_STANDING] > 0.5, torch.zeros_like(v), v)
        return torch.where(cmd[..., CMD_RANDOM] > 0.5, cmd[..., CMD_RANDOM_VEL], v)

    def proprio(self, S, a_prev, cmd) -> torch.Tensor:
        """(..., PROPRIO_DIM) noise-free, scaled frame of the proprioceptive policy terms at ``S`` (``a_prev``: the last
        action, the one that led to ``S``)."""
        p, R, v, w, q, qd = S
        frame = torch.cat(
            (
                mv(R.transpose(-1, -2), w),
                -R[..., 2, :],  # R^T (0, 0, -1)
                self.velocity_command(S, cmd),
                q - self.default_q,
                qd - self.default_qd,
                a_prev,
            ),
            -1,
        )
        return frame * self.proprio_scale

    def add_proprio_noise(self, o: torch.Tensor) -> torch.Tensor:
        """The policy group's observation noise (uniform, applied before the scale in the env)."""
        return o + (2 * torch.rand_like(o) - 1) * self.proprio_noise

    def policy_obs(self, o_hist: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Flat ``policy`` observation in the env's order and flattening. ``o_hist``: (B, HIST, PROPRIO_DIM) oldest first;
        ``depth``: (B, *DEPTH_SHAPE)."""
        hist = [o_hist[..., a:b].flatten(1) for a, b in PROPRIO_TERMS.values()]
        return torch.cat((*hist, depth.flatten(1)), -1)

    def amp_frame(self, S) -> torch.Tensor:
        """(..., AMP_FRAME_DIM) one frame of the ``amp_policy`` terms at ``S`` (no noise)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (-R[..., 2, :], q - self.default_q, (qd - self.default_qd) * self.amp_vel_scale, mv(Rt, v), mv(Rt, w)), -1
        )

    def amp_window(self, frames: torch.Tensor) -> torch.Tensor:
        """Flat ``amp_policy`` observation from (B, AMP_HIST, AMP_FRAME_DIM) frames (oldest first), in the env's order."""
        return torch.cat([frames[..., a:b].flatten(1) for a, b in AMP_TERMS.values()], -1)

    def scan(self, S) -> torch.Tensor:
        """(..., scan_dim) terrain height scan around the root (world-model input), see ``HeightScanner.height_scan``."""
        p, R = S[0], S[1]
        lead = p.shape[:-1]
        out = self.scanner.height_scan(p.reshape(-1, 3), R.reshape(-1, 3, 3))
        return out.reshape(*lead, -1)

    # ---------------- rewards ----------------
    def advance_feet_timers(self, timers: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
        """Feet air / contact timers one env step later, given the (predicted) feet contact at the new state."""
        c = contact[..., :2]
        air, cont = timers[..., 0:2], timers[..., 2:4]
        return torch.cat(((air + self.dt) * (1 - c), (cont + self.dt) * c), -1)

    def _torque(self, q, qd, a) -> torch.Tensor:
        """Applied PD torque (clamped to the effort limit) of the position + velocity actuators."""
        tau = self.kp * (self.default_q + self.act_scale * a - q) - self.kd * qd
        return torch.maximum(torch.minimum(tau, self.effort_limit), -self.effort_limit)

    def reward(self, S0, S1, a, a_prev, cmd, timers1, contact1, terms: bool = False):
        """Per-step reward of the transition ``S0 -> S1`` (action ``a``, previous action ``a_prev``), with the feet timers
        and contacts at ``S1``: the sum of all weighted, dt-scaled terms of the ``rewards`` group except
        ``volume_points_penetration`` (see below). Returns ``r`` (B,) or ``(r, {name: weighted term})``.

        ``cmd`` are the command parameters at ``S0``. Timing (measured): the env evaluates the reward BEFORE the command
        manager updates, so the velocity command is the one of the PRE-step state ``S0``; the velocity / pose of the base
        and feet are those of ``S1``; the ``feet_at_plane`` rays were cast by the previous ``sense()``, i.e. from the feet
        of ``S0``. ``dof_acc_l2`` is approximated by ((qd1 - qd0) / dt)^2 (the env uses the simulator's last-substep ``qacc``).
        """
        p, R, v, w, q, qd = S1
        qd0 = S0[5]
        w_ = self.w
        dt = self.dt
        vc = self.velocity_command(S0, cmd)
        bp, bR, bv, bw = self.kin.forward(*S1)
        v_b, w_b = mv(R.transpose(-1, -2), v), mv(R.transpose(-1, -2), w)
        c = contact1[..., :2]
        tau = self._torque(q, qd, a)
        P = self.params
        t = {}
        std = P["track_lin_vel_xy_exp"]["std"]
        t["track_lin_vel_xy_exp"] = torch.exp(-((vc[..., :2] - v_b[..., :2]) ** 2).sum(-1) / std**2)
        std = P["track_ang_vel_z_exp"]["std"]
        t["track_ang_vel_z_exp"] = torch.exp(-((vc[..., 2] - w_b[..., 2]) ** 2) / std**2)
        t["heading_error"] = vc[..., 2].abs()
        temperature = P["dont_wait"].get("velocity_temperature")
        if temperature is not None:
            t["dont_wait"] = smooth_dont_wait_penalty(
                v_b[..., 0], vc[..., 0], temperature,
                P["dont_wait"].get("command_min", 0.1), P["dont_wait"].get("command_full", 0.3),
            )
        else:
            t["dont_wait"] = (vc[..., 0] > 0.3).float() * (
                (v_b[..., 0] < 0.15).float() + (v_b[..., 0] < 0.0).float() + (v_b[..., 0] < -0.15).float()
            )
        t["is_alive"] = torch.ones_like(vc[..., 0])  # the algorithm multiplies by the survival probability
        thr = self._param("stand_still", "threshold")
        dof_error = (q - self.default_q).abs().sum(-1)
        t["stand_still"] = (
            (dof_error - P["stand_still"]["offset"])
            * (torch.linalg.vector_norm(vc[..., :2], dim=-1) < thr).float()
            * (vc[..., 2].abs() < thr).float()
        )
        # feet_air_time: soft single stance, in-mode time of each foot, min over the feet, gated by the command
        air, cont = timers1[..., 0:2], timers1[..., 2:4]
        single = c[..., 0] * (1 - c[..., 1]) + c[..., 1] * (1 - c[..., 0])
        in_mode = c * cont + (1 - c) * air
        vel_thr = P["feet_air_time"]["vel_threshold"]
        gate = ((torch.linalg.vector_norm(vc[..., :2], dim=-1) > vel_thr) | (vc[..., 2].abs() > vel_thr)).float()
        t["feet_air_time"] = single * in_mode.amin(-1) * gate
        feet_v = bv[..., K_LEFT : K_RIGHT + 1, :2]
        t["feet_slide"] = (_norm(feet_v) * c).sum(-1)
        t["joint_deviation_hip"] = ((q - self.default_q)[..., self.joint_ids["joint_deviation_hip"]] ** 2).sum(-1)
        t["ang_vel_xy_l2"] = (w_b[..., :2] ** 2).sum(-1)
        t["dof_torques_l2"] = (tau[..., self.joint_ids["dof_torques_l2"]] ** 2).sum(-1)
        t["dof_acc_l2"] = (((qd - qd0) / dt) ** 2).sum(-1)
        t["dof_vel_l2"] = (qd**2).sum(-1)
        t["action_rate_l2"] = ((a - a_prev) ** 2).sum(-1)
        t["flat_orientation_l2"] = (R[..., 2, :2] ** 2).sum(-1)
        t["pelvis_orientation_l2"] = (bR[..., K_PELVIS, 2, :2] ** 2).sum(-1)
        feet_R = bR[..., K_LEFT : K_RIGHT + 1, :, :]
        t["feet_flat_ori"] = (_norm(feet_R[..., 2, :2]) * c).sum(-1)
        # feet_at_plane: two rays per foot at x = +-0.06 m in the foot yaw frame, cast from the feet of S0
        bp0, bR0 = self.kin.forward(*S0)[:2]
        feet_p0 = bp0[..., K_LEFT : K_RIGHT + 1, :]
        col = yaw_mat(bR0[..., K_LEFT : K_RIGHT + 1, :, :])[..., :2, :2]  # (B, 2, 2, 2)
        ray_xy = feet_p0[..., None, :2] + mv(col.unsqueeze(-3), self.foot_rays)  # (B, feet, rays, 2)
        gz = self.scanner.ground_height(ray_xy.flatten(-3, -2)).reshape(ray_xy.shape[:-1])
        hit_z = torch.where(feet_p0[..., None, 2] - gz > self.ray_max_distance, torch.zeros_like(gz), gz)  # miss -> 0
        off = self._param("feet_at_plane", "height_offset")
        h = torch.clamp(bp[..., K_LEFT : K_RIGHT + 1, None, 2] - hit_z - off, min=0.0, max=0.3)
        t["feet_at_plane"] = (h * c[..., None]).sum((-2, -1))
        heading = _norm(R[..., :2, 0])
        cos_h, sin_h = R[..., 0, 0] / heading, R[..., 1, 0] / heading
        feet_y = -sin_h[..., None] * bp[..., K_LEFT : K_RIGHT + 1, 0] + cos_h[..., None] * bp[..., K_LEFT : K_RIGHT + 1, 1]
        dist_y = (feet_y[..., 0] - feet_y[..., 1]).abs()
        t["feet_close_xy"] = torch.exp(-torch.clamp(P["feet_close_xy"]["threshold"] - dist_y, min=0.0) / P["feet_close_xy"]["std"] ** 2) - 1
        power = (tau * qd / self.kp)[..., self.joint_ids["energy"]]  # normalize_by_stiffness
        t["energy"] = (power**2).sum(-1)
        t["freeze_upper_body"] = (q - self.default_q)[..., self.joint_ids["freeze_upper_body"]].abs().sum(-1)
        t["dof_pos_limits"] = (torch.relu(self.soft_lo - q) + torch.relu(q - self.soft_hi)).sum(-1)
        t["dof_vel_limits"] = torch.clamp(qd.abs() - self.vel_limits * P["dof_vel_limits"]["soft_ratio"], 0.0, 1.0).sum(-1)
        limit = self.effort_limit * P["torque_limits"]["limit_ratio"]
        t["torque_limits"] = (torch.relu(tau.abs() - limit) ** 2).sum(-1)
        t["undesired_contacts"] = contact1[..., 3:].sum(-1)
        t = {k: t[k] * w_[k] * dt for k in w_ if k != "volume_points_penetration"}
        r = sum(t.values())
        return (r, t) if terms else r

    @torch.compiler.disable
    def volume_points_penetration(self, S1) -> torch.Tensor:
        """Weighted, dt-scaled ``volume_points_penetration`` term (B,): exact value of the env's virtual edge-cylinder
        penetration of the leg volume points at ``S1``; gradient first order along the penetration direction."""
        bp, bR, bv, bw = self.kin.forward(*S1)
        sl = slice(K_LEFT, K_RIGHT + 1)
        pos, rot, lin, ang = bp[:, sl], bR[:, sl], bv[:, sl], bw[:, sl]
        # subtree COMs: whole robot (the reference point of ``cvel``) and the link itself (used by the sensor)
        com_all = ((bp * self.com_mass[:, None] + mv(bR, self.com_moment)).sum(1)) / self.com_mass.sum()
        com_link = pos + mv(rot, self.com_moment[sl] / self.com_mass[sl, None])
        lin = lin + torch.linalg.cross(ang, com_all.unsqueeze(1) - com_link, dim=-1)  # the sensor's (shifted) link velocity
        pts = mv(rot.unsqueeze(-3), self.vp_pattern) + pos.unsqueeze(-2)  # (B, 2, P, 3), world
        vel = lin.unsqueeze(-2) + torch.linalg.cross(ang.unsqueeze(-2), pts - pos.unsqueeze(-2), dim=-1)
        # penetration offsets exactly as ``VolumePoints._refresh_penetration_offset`` (no grad)
        flat = pts.detach().reshape(-1, 3).contiguous()
        offset_buf = torch.zeros_like(flat)
        depth_buf = torch.zeros_like(flat[:, 0])
        for obstacle in self.vp_sensor._virtual_obstacles.values():
            offset = obstacle.get_points_penetration_offset(flat)
            depth = torch.norm(offset, dim=-1)
            mask = depth > depth_buf
            depth_buf[mask] = depth[mask]
            offset_buf[mask] = offset[mask]
        depth0 = depth_buf.reshape(pts.shape[:-1])
        direction = offset_buf.reshape(pts.shape) / depth0.clamp(min=1e-12).unsqueeze(-1)
        depth = depth0 + ((pts - pts.detach()) * direction).sum(-1)
        tol = self._param("volume_points_penetration", "tolerance")
        in_obstacle = (depth0 > tol).float()
        val = (in_obstacle * (torch.linalg.vector_norm(vel, dim=-1) + 1e-6) * depth).flatten(1).sum(-1)
        return val * self.w["volume_points_penetration"] * self.dt

    # ---------------- terminations / critic ----------------
    def _tilt(self, R: torch.Tensor) -> torch.Tensor:
        """acos(-projected_gravity_z) = acos(R_zz), smooth."""
        return torch.atan2(_norm(R[..., 2, :2]), R[..., 2, 2])

    def hard_termination(self, S, cmd, contact) -> torch.Tensor:
        """(B,) bool: ``base_contact`` | ``bad_orientation`` | ``root_height`` (the time-out terms are not terminations)."""
        p, R = S[0], S[1]
        base_contact = contact[..., 2] > 0.5
        bad_orientation = torch.acos(torch.clamp(R[..., 2, 2], -1.0, 1.0)).abs() > self.limit_angle
        low = p[..., 2] - torch.clamp(cmd[..., CMD_ORIGIN][..., 2], max=0.0) < self.min_height
        return base_contact | bad_orientation | low

    def soft_termination(self, S, cmd, contact, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """(B,) differentiable probability version of ``hard_termination``."""
        p, R = S[0], S[1]
        height = p[..., 2] - torch.clamp(cmd[..., CMD_ORIGIN][..., 2], max=0.0)
        alive = 1 - contact[..., 2]
        alive = alive * (1 - torch.sigmoid((self._tilt(R) - self.limit_angle) / tt))
        alive = alive * (1 - torch.sigmoid((self.min_height - height) / tz))
        return 1 - alive

    def critic_features(self, S, cmd, timers, xi_n) -> torch.Tensor:
        """(..., critic_dim) privileged critic / failure-head features (no noise)."""
        p, R = S[0], S[1]
        t_b = mv(yaw_mat(R).transpose(-1, -2), cmd[..., CMD_TARGET] - p)
        target = torch.cat((t_b[..., :2], _norm(t_b[..., :2]).unsqueeze(-1)), -1)
        return torch.cat(
            (
                invariant_features(S),
                self.velocity_command(S, cmd),
                target,
                cmd[..., CMD_MAX],
                cmd[..., CMD_STANDING],
                cmd[..., CMD_RANDOM],
                timers,
                xi_n,
                self.scan(S),
            ),
            -1,
        )


def build_parkour_mdp(env, heightmap_resolution: float, heightmap_cache_dir: str, device: str):
    """FK, heightmap scanner and analytic MDP for the live (unwrapped) parkour env: ``(kin, scanner, mdp)``."""
    joint_names = list(env.scene["robot"].joint_names)
    names = G1Kinematics(device, joint_names, TRACKED).body_names
    kin = G1Kinematics(device, joint_names, TRACKED + [n for n in names if n not in TRACKED])  # all bodies: COM tables
    heights, x0, y0 = load_or_build_heightmap(env.sim.mj_model, None, heightmap_resolution, heightmap_cache_dir)
    scanner = HeightScanner(
        heights, x0, y0, heightmap_resolution, HEIGHT_SCANNER_SIZE, HEIGHT_SCANNER_RESOLUTION, HEIGHT_SCANNER_MAX_DISTANCE, device
    )
    return kin, scanner, ParkourMDP(env, kin, scanner, device)
