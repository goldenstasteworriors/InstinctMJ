"""GPU ring buffer with episode-boundary masks and chronological history."""

from __future__ import annotations

import torch


class ReplayBuffer:
  def __init__(self, steps: int, envs: int, device: str):
    self.steps, self.envs, self.device = steps, envs, device
    self.storage: dict[str, torch.Tensor] = {}
    self.total = 0

  @torch.no_grad()
  def add(self, **values: torch.Tensor) -> None:
    if not self.storage:
      for key, value in values.items():
        self.storage[key] = torch.empty((self.steps, *value.shape), dtype=value.dtype, device=self.device)
    for key, value in values.items():
      self.storage[key][self.total % self.steps].copy_(value)
    self.total += 1

  def sample(self, batch: int, window: int = 1, history: int = 10) -> dict[str, torch.Tensor]:
    oldest = max(0, self.total - self.steps)
    lo, hi = oldest + history - 1, self.total - window + 1
    if hi <= lo:
      raise ValueError("Not enough chronological replay for the requested history/window")
    t = torch.randint(lo, hi, (batch,), device=self.device)
    e = torch.randint(self.envs, (batch,), device=self.device)
    ts = (t[:, None] + torch.arange(window, device=self.device)) % self.steps
    output = {k: v[ts, e[:, None]] for k, v in self.storage.items()}
    ht = (t[:, None] + torch.arange(1 - history, 1, device=self.device)) % self.steps
    output["history"] = self.storage["state"][ht, e[:, None]]
    # Replace history preceding the most recent reset with the episode's initial frame.
    hd = self.storage["done"][ht, e[:, None]]
    indices = torch.arange(history, device=self.device).expand(batch, -1)
    last_reset = torch.where(hd[:, :-1], indices[:, 1:], 0).amax(-1)
    gather = torch.maximum(indices, last_reset[:, None])
    output["history"] = output["history"].gather(1, gather[..., None].expand_as(output["history"]))
    done = output["done"]
    output["mask"] = torch.cat((torch.ones(batch, 1, device=self.device), (~done[:, :-1]).float().cumprod(1)), 1)
    return output
