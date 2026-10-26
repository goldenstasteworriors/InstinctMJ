"""Differentiable rigid-body kinematics and perceptive tracking networks.

Physical state: root xyz/quaternion, 29 q, root world v/w, 29 qdot.
The remaining state is last action, 18x32 depth, log contact-history maxima,
and applied joint torque. Terrain and randomized physics are separate context.
"""

import torch
import torch.nn.functional as F
from torch import nn

from instinct_mj.supermimic.networks import RunningNorm, mlp

PHYS = 71
DEPTH = slice(100, 676)
CONTACT = slice(676, 716)
TORQUE = slice(716, 745)
STATE = 745
REF_DIM = 254


def mul(a, b):
  aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
  return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
    aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def inv(q):
  return torch.cat((q[..., :1], -q[..., 1:]), -1)


def rotate_full(q, v):
  xyz, v = torch.broadcast_tensors(q[..., 1:], v)
  uv = 2 * torch.cross(xyz, v, dim=-1)
  return v + q[..., :1] * uv + torch.cross(xyz, uv, dim=-1)


def angle_squared(a, b):
  d = mul(a, inv(b))
  n = d[..., 1:].norm(dim=-1)
  return (2 * torch.atan2(n, d[..., 0].abs().clamp_min(1e-8))).square()


def yaw(q):
  w, x, y, z = q.unbind(-1)
  a = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) / 2
  zero = torch.zeros_like(a)
  return torch.stack((a.cos(), zero, zero, a.sin()), -1)


class Kinematics(nn.Module):
  """Batched tree FK including exact link-origin linear/angular velocities."""
  def __init__(self, model, robot, device):
    super().__init__()
    body_ids = robot.indexing.body_ids.tolist()
    joint_ids = robot.indexing.joint_ids.tolist()
    self.names = list(robot.body_names)
    self.groups = []
    parents, levels, joints = [], [], []
    for i, bid in enumerate(body_ids):
      parent = int(model.body_parentid[bid])
      parents.append(body_ids.index(parent) if parent in body_ids else 0)
      levels.append(0 if i == 0 else levels[parents[-1]] + 1)
      owned = [j for j in joint_ids if int(model.jnt_bodyid[j]) == bid]
      if len(owned) > 1:
        raise ValueError("FK expects at most one hinge per body")
      joints.append(joint_ids.index(owned[0]) if owned else -1)
    self.register_buffer("parent", torch.tensor(parents, device=device))
    self.register_buffer("joint", torch.tensor(joints, device=device))
    for name, array in (("offset", model.body_pos[body_ids]), ("rotation", model.body_quat[body_ids]),
                        ("axis", model.jnt_axis[joint_ids]), ("pivot", model.jnt_pos[joint_ids]),
                        ("qzero", model.qpos0[model.jnt_qposadr[joint_ids]])):
      self.register_buffer(name, torch.tensor(array.copy(), dtype=torch.float32, device=device))
    for level in range(1, max(levels) + 1):
      name = f"level_{level}"
      self.register_buffer(name, torch.tensor([i for i, d in enumerate(levels) if d == level], device=device))
      self.groups.append(name)

  def forward(self, s):
    n, b = len(s), len(self.names)
    p = s[:, None, :3].expand(n, b, 3)
    q = s[:, None, 3:7].expand(n, b, 4)
    v = s[:, None, 36:39].expand(n, b, 3)
    w = s[:, None, 39:42].expand(n, b, 3)
    for name in self.groups:
      ids = getattr(self, name)
      par, j = self.parent[ids], self.joint[ids]
      valid, j = (j >= 0).float()[None, :, None], j.clamp_min(0)
      pq = q[:, par]
      pre_q = mul(pq, self.rotation[ids][None].expand(n, -1, -1))
      pre_p = p[:, par] + rotate_full(pq, self.offset[ids][None])
      a = (s[:, 7:36][:, j] - self.qzero[j])[:, :, None] * valid
      dq = torch.cat(((a / 2).cos(), self.axis[j][None] * (a / 2).sin()), -1)
      out_q = mul(pre_q, dq)
      pivot = self.pivot[j][None] * valid
      joint_w = rotate_full(pre_q, self.axis[j][None]) * s[:, 42:71][:, j, None] * valid
      joint_p = pre_p + rotate_full(pre_q, pivot)
      out_p = pre_p + rotate_full(pre_q, pivot) - rotate_full(out_q, pivot)
      out_v = v[:, par] + torch.cross(w[:, par], out_p - p[:, par], dim=-1)
      out_v = out_v + torch.cross(joint_w, out_p - joint_p, dim=-1)
      p, q = p.index_copy(1, ids, out_p), q.index_copy(1, ids, out_q)
      v, w = v.index_copy(1, ids, out_v), w.index_copy(1, ids, w[:, par] + joint_w)
    return p, q, v, w


def proprio(s, defaults, noise=False):
  gravity = torch.zeros_like(s[..., :3])
  gravity[..., 2] = -1
  obs = torch.cat((rotate_full(inv(s[..., 3:7]), s[..., 39:42]),
    rotate_full(inv(s[..., 3:7]), gravity), s[..., 7:36] - defaults,
    s[..., 42:71] * 0.05, s[..., 71:100]), -1)
  if noise:
    scale = obs.new_tensor([0.2] * 3 + [0.05] * 3 + [0.01] * 29 + [0.025] * 29 + [0.0] * 29)
    obs = obs + (2 * torch.rand_like(obs) - 1) * scale
  return obs


def reference_at(ref, step):
  """Interpolate the logged exogenous 10 Hz reference, never predict its future."""
  t = torch.as_tensor(step, device=ref.device, dtype=torch.float32) / 5
  lo = t.floor().long().clamp(0, ref.shape[1] - 2)
  f = (t - lo).clamp(0, 1)
  a, b = ref[:, lo].float(), ref[:, lo + 1].float()
  f = f.reshape(1, -1, 1) if t.ndim else f
  out = a * (1 - f) + b * f
  # Root and body quaternion interpolation follows the shortest hemisphere.
  root_b = torch.where((a[..., 3:7] * b[..., 3:7]).sum(-1, keepdim=True) < 0, -b[..., 3:7], b[..., 3:7])
  root_q = F.normalize(a[..., 3:7] * (1 - f) + root_b * f, dim=-1)
  aq, bq = a[..., 113:169].reshape(*a.shape[:-1], 14, 4), b[..., 113:169].reshape(*b.shape[:-1], 14, 4)
  bq = torch.where((aq * bq).sum(-1, keepdim=True) < 0, -bq, bq)
  fq = f.unsqueeze(-1) if t.ndim else f
  body_q = F.normalize(aq * (1 - fq) + bq * fq, dim=-1).flatten(-2)
  return torch.cat((out[..., :3], root_q, out[..., 7:113], body_q, out[..., 169:]), -1)


def goal(s, future, defaults):
  q = s[:, None, 3:7].expand(-1, 10, -1)
  r = mul(inv(q), future[..., 3:7])
  x = torch.zeros_like(r[..., :3]); x[..., 0] = 1
  z = torch.zeros_like(x); z[..., 2] = 1
  refq = future[:, :1, 3:7].expand(-1, 10, -1)
  pos = rotate_full(inv(refq), future[..., :3] - future[:, :1, :3])
  return torch.cat((future[..., 7:36] - defaults[:, None], future[..., 42:71] * 0.05,
    pos, rotate_full(r, x), rotate_full(r, z)), -1).flatten(1)


class TrackingPolicy(nn.Module):
  def __init__(self):
    super().__init__()
    self.obs_norm, self.goal_norm = RunningNorm(93), RunningNorm(670)
    self.encoder = mlp(670, [1024, 512, 256], 32)
    self.vision = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ELU(),
      nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(32 * 5 * 8, 64), nn.ELU())
    self.decoder = mlp(930 + 32 + 64, [1024, 512, 256], 29)
    with torch.no_grad():
      self.decoder[-1].weight.mul_(0.01)
      self.decoder[-1].bias.zero_()

  def forward(self, history, s, future, defaults):
    g = torch.tanh(self.encoder(self.goal_norm(goal(s, future, defaults))))
    d = self.vision(s[:, DEPTH].reshape(-1, 1, 18, 32))
    return self.decoder(torch.cat((self.obs_norm(history).flatten(1), g, d), -1)).clamp(-10, 10)


class TrackingWorldModel(nn.Module):
  def __init__(self, context_dim, width=1024):
    super().__init__()
    self.input_norm = RunningNorm(STATE + 29 + context_dim, floor=0.01)
    self.delta_norm = RunningNorm(STATE, floor=0.02)
    self.net = mlp(STATE + 29 + context_dim, [width] * 3, STATE, True)
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, s, a, context):
    delta = self.net(self.input_norm(torch.cat((s, a, context), -1)))
    nxt = s + delta * self.delta_norm.std() + self.delta_norm.mean
    return torch.cat((nxt[:, :3], F.normalize(nxt[:, 3:7], dim=-1), nxt[:, 7:71], a,
      nxt[:, DEPTH].clamp(-1, 1), nxt[:, CONTACT].clamp(0, 12), nxt[:, TORQUE]), -1)


class TrackingCritic(nn.Module):
  def __init__(self, context_dim):
    super().__init__()
    self.norm = RunningNorm(STATE + PHYS + context_dim, floor=0.02)
    self.net = mlp(STATE + PHYS + context_dim, [512, 256], 1)

  def forward(self, s, r, ctx):
    return self.net(self.norm(torch.cat((s, r[..., :PHYS], ctx), -1))).squeeze(-1)
