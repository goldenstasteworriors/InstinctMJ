"""Smooth reward primitives shared by simulator managers and imagined rollouts."""

from __future__ import annotations

import torch


def smooth_dont_wait_penalty(
    forward_velocity: torch.Tensor,
    forward_command: torch.Tensor,
    velocity_temperature: float = 0.05,
    command_min: float = 0.1,
    command_full: float = 0.3,
) -> torch.Tensor:
    """Bounded [0, 3] forward-motion penalty; apply a negative reward weight.

    Velocities are body-frame m/s. Sigmoids replace the original speed tests at
    0.15, 0 and -0.15 m/s. A C2 smoothstep command gate is exactly zero below
    ``command_min`` and one above ``command_full``: standing, turning in place
    and a reached target must not create an incentive to move forward.
    """
    if velocity_temperature <= 0 or command_full <= command_min:
        raise ValueError("Require positive velocity_temperature and command_full > command_min")
    x = ((forward_command - command_min) / (command_full - command_min)).clamp(0.0, 1.0)
    gate = x.pow(3) * (10.0 + x * (-15.0 + 6.0 * x))
    penalty = (
        torch.sigmoid((0.15 - forward_velocity) / velocity_temperature)
        + torch.sigmoid(-forward_velocity / velocity_temperature)
        + torch.sigmoid((-0.15 - forward_velocity) / velocity_temperature)
    )
    return gate * penalty
