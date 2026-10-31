"""Synthetic temporal negatives for AMP; no simulator or world-model input changes."""
import torch
from .mdp import AMP_HIST, AMP_TERMS


@torch.no_grad()
def moving_reference_negatives(reference: torch.Tensor) -> torch.Tensor:
    """Freeze half and permute half of moving windows (mean body XY speed >0.2m/s).

Keep the original reference positives. Standing reference windows are excluded.
Permuted velocities are intentionally not derivatives of permuted poses: these
are invalid-temporal-order stress examples, not physical trajectories.
"""
    lo, hi = AMP_TERMS["base_lin_vel"]
    velocity = reference[:, lo * AMP_HIST:hi * AMP_HIST].reshape(-1, AMP_HIST, hi - lo)
    moving = velocity[..., :2].norm(dim=-1).mean(-1) > 0.2
    reference = reference[moving]
    n = len(reference)
    if n == 0:
        return reference
    static = torch.rand(n, device=reference.device) < 0.5
    order = torch.rand(n, AMP_HIST, device=reference.device).argsort(-1)
    parts = []
    for name, (lo, hi) in AMP_TERMS.items():
        x = reference[:, lo * AMP_HIST:hi * AMP_HIST].reshape(n, AMP_HIST, hi - lo)
        frozen = torch.zeros_like(x) if "vel" in name else x[:, -1:].expand_as(x)
        permuted = x.gather(1, order[:, :, None].expand_as(x))
        parts.append(torch.where(static[:, None, None], frozen, permuted).flatten(1))
    return torch.cat(parts, -1)
