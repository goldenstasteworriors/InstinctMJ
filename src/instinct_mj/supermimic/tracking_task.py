"""Native task extraction and analytic tracking objectives, with runtime audits."""

import torch
import torch.nn.functional as F

from instinct_mj.supermimic.tracking_model import (
  CONTACT,
  TORQUE,
  Kinematics,
  angle_squared,
  inv,
  mul,
  rotate_full,
  yaw,
)


class TrackingTask:
  def __init__(self, env):
    self.env, self.robot = env, env.scene["robot"]
    self.motion = env.scene["motion_reference"]
    self.buffer = next(iter(self.motion.motion_buffers.values()))
    self.fk = Kinematics(env.sim.mj_model, self.robot, env.device)
    self.links = self.robot.find_bodies(self.motion.cfg.link_of_interests, preserve_order=True)[0]
    self.extremities = [self.motion.cfg.link_of_interests.index(n) for n in
      ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")]
    self.defaults = self.robot.data.default_joint_pos.clone()
    contact_cfg = env.reward_manager.get_term_cfg("undesired_contacts").params["asset_cfg"]
    self.penalty_bodies = self.robot.find_bodies(contact_cfg.body_names)[0]
    self.torque_joints = self.robot.find_joints([".*ankle.*", ".*wrist.*"])[0]
    model = env.sim.mj_model
    self.limits = self.robot.data.soft_joint_pos_limits.clone()
    self.effort = torch.zeros(29, device=env.device)
    for act in self.robot.actuators:
      for local, ctrl in zip(act.target_ids, act.global_ctrl_ids):
        self.effort[int(local)] = float(abs(model.actuator_forcerange[int(ctrl)]).max())
    context = [self.defaults]
    # Full randomized arrays, rather than averages that hide per-link parameters.
    specs = (("geom_friction", self.robot.indexing.geom_ids, 2),
      ("body_mass", self.robot.indexing.body_ids, 1),
      ("body_ipos", self.robot.indexing.body_ids, 2),
      ("actuator_gainprm", self.robot.indexing.ctrl_ids, 2),
      ("actuator_biasprm", self.robot.indexing.ctrl_ids, 2))
    for name, idx, base_ndim in specs:
      value = getattr(env.sim.model, name)
      if value.ndim == base_ndim:
        value = value.unsqueeze(0).expand(env.num_envs, *value.shape)
      elif value.shape[0] == 1:
        value = value.expand(env.num_envs, *value.shape[1:])
      context.append(value[:, idx].flatten(1))
    self.physics = torch.cat(context, -1).clone()
    self.num_terrains = len(self.buffer.yaml_data["terrains"])
    self.term_names = list(env.reward_manager._termwise_reward_buf["rewards"])

  def origins(self):
    return self.buffer._assigned_env_origins

  def context(self):
    motion_ids = self.buffer._assigned_env_motion_selection.long()
    terrain_ids = self.buffer._all_motion_terrain_ids[motion_ids].long()
    return torch.cat((self.physics, F.one_hot(terrain_ids, self.num_terrains).float()), -1)

  def state(self, obs):
    d = self.robot.data
    c = self.env.scene["contact_forces"].data.force_history.norm(dim=-1).amax(-1)
    s = torch.cat((d.root_link_pos_w - self.origins(), d.root_link_quat_w, d.joint_pos,
      d.root_link_lin_vel_w, d.root_link_ang_vel_w, d.joint_vel,
      self.env.action_manager.action, obs["policy"]["depth_image"].reshape(self.env.num_envs, -1),
      torch.log1p(c), d.qfrc_actuator), -1)
    if s.shape[1] != 745 or not torch.isfinite(s).all():
      raise ValueError(f"Invalid tracking state: {s.shape}")
    return s

  def reference(self, current=False):
    d = self.motion.reference_frame if current else self.motion.data
    origin = self.origins()[:, None]
    # MotionReferenceManager already exposes joint tensors in simulation order.
    return torch.cat((d.base_pos_w - origin, d.base_quat_w, d.joint_pos,
      d.base_lin_vel_w, d.base_ang_vel_w, d.joint_vel,
      (d.link_pos_w - origin[:, :, None]).flatten(2), d.link_quat_w.flatten(2),
      d.link_lin_vel_w.flatten(2), d.link_ang_vel_w.flatten(2), d.validity[..., None].float()), -1)

  def reward_terms(self, previous, s, action, ref, limits, smooth=False, with_continuation=False):
    all_p, q, v, w = self.fk(s)
    p, q, v, w = all_p[:, self.links], q[:, self.links], v[:, self.links], w[:, self.links]
    rp, rq = ref[:, 71:113].reshape(-1, 14, 3), ref[:, 113:169].reshape(-1, 14, 4)
    rv, rw = ref[:, 169:211].reshape(-1, 14, 3), ref[:, 211:253].reshape(-1, 14, 3)
    delta = yaw(mul(s[:, 3:7], inv(ref[:, 3:7])))[:, None].expand(-1, 14, -1)
    relative_base = torch.cat((s[:, :2], ref[:, 2:3]), -1)
    aligned_p = relative_base[:, None] + rotate_full(delta, rp - ref[:, None, :3])
    aligned_q = mul(delta, rq)
    terms = {
      "base_position_imitation_gauss": 0.5 * torch.exp(-(s[:, :3] - ref[:, :3]).square().sum(-1) / 0.3**2) * ref[:, 253],
      "base_rot_imitation_gauss": 0.5 * torch.exp(-angle_squared(s[:, 3:7], ref[:, 3:7]) / 0.4**2),
      "link_pos_imitation_gauss": torch.exp(-(p - aligned_p).square().sum(-1).mean(-1) / 0.3**2),
      "link_rot_imitation_gauss": torch.exp(-angle_squared(q, aligned_q).mean(-1) / 0.4**2),
      "link_lin_vel_imitation_gauss": torch.exp(-(v - rv).square().sum(-1).mean(-1)),
      "link_ang_vel_imitation_gauss": torch.exp(-(w - rw).square().sum(-1).mean(-1) / 3.14**2),
      "action_rate_l2": -0.1 * (action - previous[:, 71:100]).square().sum(-1),
      "joint_limit": -10 * ((limits[..., 0] - s[:, 7:36]).clamp_min(0) +
        (s[:, 7:36] - limits[..., 1]).clamp_min(0)).sum(-1),
      "applied_torque_limits_by_ratio": -0.05 * (s[:, TORQUE][:, self.torque_joints].abs() -
        self.effort[self.torque_joints] * 0.8).clamp_min(0).square().sum(-1),
    }
    c = s[:, CONTACT][:, self.penalty_bodies]
    contact = torch.sigmoid((c - 0.69314718056) / 0.1) if smooth else (c > 0.69314718056).float()
    terms["undesired_contacts"] = -0.1 * contact.sum(-1)
    result = torch.stack([terms[k] for k in self.term_names], -1) * self.env.step_dt
    if with_continuation:
      return result, self._continuation(s, ref, p)
    return result

  def continuation(self, s, ref):
    p, _, _, _ = self.fk(s)
    return self._continuation(s, ref, p[:, self.links])

  def _continuation(self, s, ref, p):
    rp = ref[:, 71:113].reshape(-1, 14, 3)
    gravity = torch.zeros_like(s[:, :3]); gravity[:, 2] = -1
    pg = (rotate_full(inv(s[:, 3:7]), gravity) - rotate_full(inv(ref[:, 3:7]), gravity)).norm(dim=-1)
    root = (s[:, 2] - ref[:, 2]).abs()
    links = (p[:, self.extremities, 2] - rp[:, self.extremities, 2]).abs().amax(-1)
    return torch.sigmoid((0.25 - root) / 0.025) * torch.sigmoid((0.8 - pg) / 0.08) * torch.sigmoid((0.25 - links) / 0.025)

  @torch.no_grad()
  def audit(self, s):
    # Native rewards cache these during pre-integration reward computation.
    # Recompute against the current post-forward robot pose for an exact audit.
    for name in ("_reference_relative_base_pos", "_reference_relative_delta_quat",
                 "_reference_link_pos_relative_w", "_reference_link_quat_relative_w"):
      buffer = getattr(self.motion, name)
      buffer.timestamp = torch.full_like(buffer.timestamp, -1)
    p, q, v, w = self.fk(s)
    d = self.robot.data
    errors = {
      "fk_position_max_m": float((p - (d.body_link_pos_w - self.origins()[:, None])).abs().max()),
      "fk_rotation_max_rad": float(angle_squared(q, d.body_link_quat_w).sqrt().max()),
      "fk_velocity_max_mps": float((v - d.body_link_lin_vel_w).abs().max()),
      "fk_angular_velocity_max_radps": float((w - d.body_link_ang_vel_w).abs().max()),
    }
    if max(errors.values()) > 0.002:
      raise AssertionError(f"MuJoCo/FK mismatch: {errors}")
    ref = self.reference(current=True)[:, 0]
    analytic = self.reward_terms(s, s, s[:, 71:100], ref, self.limits)
    errors["native_reward_max_abs"] = 0.0
    for i, name in enumerate(self.term_names):
      if name == "action_rate_l2":
        continue  # Compared on actual transitions by the trainer.
      cfg = self.env.reward_manager.get_term_cfg(name)
      native = cfg.func(self.env, **cfg.params) * cfg.weight * self.env.step_dt
      error = float((native - analytic[:, i]).abs().max())
      errors[f"reward_{name}_max_abs"] = error
      errors["native_reward_max_abs"] = max(errors["native_reward_max_abs"], error)
    if errors["native_reward_max_abs"] > 2e-4:
      raise AssertionError(f"Native reward mismatch: {errors}")
    return errors
