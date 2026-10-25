"""Networks following zita-ch/supermimic_vibed, extended for perceptive parkour.

The original implementation is MIT licensed (Chong Zhang, 2026).
Parkour uses sensor-dependent rewards; this variant fits individual task reward
terms and termination alongside state dynamics instead of assuming flat ground.
"""

from __future__ import annotations

import torch
from torch import nn


def mlp(inp: int, hidden: list[int], out: int, layer_norm: bool = False) -> nn.Sequential:
  layers: list[nn.Module] = []
  for width in hidden:
    layers.append(nn.Linear(inp, width))
    if layer_norm:
      layers.append(nn.LayerNorm(width))
    layers.append(nn.SiLU() if layer_norm else nn.ELU())
    inp = width
  layers.append(nn.Linear(inp, out))
  return nn.Sequential(*layers)


class RunningNorm(nn.Module):
  def __init__(self, dim: int, floor: float = 0.01):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(()))
    self.floor = floor

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1]).float()
    n = x.shape[0]
    mean, var = x.mean(0), x.var(0, unbiased=False)
    total = self.count + n
    delta = mean - self.mean
    self.var.copy_((self.var * self.count + var * n + delta.square() * self.count * n / total) / total)
    self.mean.add_(delta * n / total)
    self.count.copy_(total)

  def std(self) -> torch.Tensor:
    return self.var.clamp_min(self.floor**2).sqrt()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std()


class WorldModel(nn.Module):
  def __init__(self, state_dim: int, action_dim: int, reward_dim: int, width: int = 1024):
    super().__init__()
    self.state_dim, self.reward_dim = state_dim, reward_dim
    self.input_norm = RunningNorm(state_dim + action_dim)
    self.delta_norm = RunningNorm(state_dim, floor=0.003)
    self.reward_norm = RunningNorm(reward_dim, floor=0.01)
    self.net = mlp(state_dim + action_dim, [width] * 3, state_dim + reward_dim + 1, True)
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
    output = self.net(self.input_norm(torch.cat((state, action), -1)))
    delta, reward, done = output.split((self.state_dim, self.reward_dim, 1), -1)
    next_state = state + delta * self.delta_norm.std() + self.delta_norm.mean
    reward = reward * self.reward_norm.std() + self.reward_norm.mean
    return next_state, reward, done.squeeze(-1)


class Policy(nn.Module):
  def __init__(self, proprio_dim: int, goal_dim: int, hist: int = 10):
    super().__init__()
    self.obs_norm = RunningNorm(proprio_dim)
    self.goal_norm = RunningNorm(goal_dim)
    self.enc = mlp(goal_dim, [2048, 1024, 512, 256], 32)
    self.dec = mlp(proprio_dim * hist + 32, [2048, 1024, 512, 256], 29)
    with torch.no_grad():
      self.dec[-1].weight.mul_(0.01)
      self.dec[-1].bias.zero_()

  def forward(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    z = torch.tanh(self.enc(self.goal_norm(goal)))
    return self.dec(torch.cat((self.obs_norm(history).flatten(-2), z), -1)).clamp(-20, 20)


class Critic(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.norm = RunningNorm(dim)
    self.net = mlp(dim, [256, 128], 1)

  def forward(self, state: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(state)).squeeze(-1)
