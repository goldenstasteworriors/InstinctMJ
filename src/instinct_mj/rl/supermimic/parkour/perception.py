"""Modality decoder and held-out perception intervention diagnostics."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mdp import unflatten_state
from ..networks import mlp


class PerceptionDecoder(nn.Module):
    def __init__(self, latent_dim, shape):
        super().__init__()
        self.shape = shape
        self.net = mlp(latent_dim, [256, 256], math.prod(shape))

    def forward(self, z):
        return self.net(z).sigmoid().reshape(-1, *self.shape)


@torch.no_grad()
def perception_diagnostics(runner, num=128):
    """Same held-out windows/actions for all interventions. Never updates models.

Shuffle/mean replace only initial perception; future latents are predicted.
Physical errors can be compared across encoders, unlike raw latent MSE.
Windows end at real episode resets, using the replay validity mask.
"""
    if runner.cfg["wm_holdout_envs"] <= 0:
        raise ValueError("Perception diagnostics require held-out environments")
    # Diagnostics must not perturb subsequent training random-number streams.
    with torch.random.fork_rng(devices=[torch.device(runner.device)]):
        torch.manual_seed(73000 + runner.it)
        w = runner.buffer.sample_windows(num, 32, runner.buffer.holdout_envs)
        depth = w["depth"][:, 0].float() / 255.0
        z0 = runner.encode_depth(depth)
        xi = runner.episode_context(w["lags"], runner.fr_n[w["env"]].unsqueeze(1).expand(-1, 32, -1))
        ctx = runner.wm_context(xi, w["a_prev"])
        # A cyclic permutation is a derangement: every sample receives another window's latent.
        zs = {"intact": z0, "shuffled": z0.roll(1, 0), "mean": runner.wm.z_mean.expand_as(z0)}
        out = {}
        span = 2.0 if runner.perception_term == "height_image" else 2.5
        for mode, latent in zs.items():
            state, z = unflatten_state(w["s"][:, 0]), latent
            for j in range(32):
                state, logits, z = runner.wm.step(state, w["a"][:, j], ctx[:, j], runner.wm_scan(state), z)
                if j + 1 not in (1, 8, 16, 32):
                    continue
                mask = w["mask"][:, j]
                denom = mask.sum().clamp_min(1)
                truth = unflatten_state(w["s1"][:, j])
                prefix = f"wm_diag/{mode}/h{j+1}"
                out[f"{prefix}/valid_windows"] = float(mask.sum())
                for name, idx in (("position_rmse_m", 0), ("velocity_rmse_mps", 2), ("joint_rmse_rad", 4)):
                    squared = (state[idx] - truth[idx]).square().mean(-1)
                    out[f"{prefix}/{name}"] = float(((squared * mask).sum() / denom).sqrt())
                bce = F.binary_cross_entropy_with_logits(logits, w["contact"][:, j], reduction="none").mean(-1)
                out[f"{prefix}/contact_bce"] = float((bce * mask).sum() / denom)
                target_image = w["depth1"][:, j].float() / 255.0
                image_error = (runner.latent_dec(z) - target_image).abs().flatten(1).mean(-1) * span
                out[f"{prefix}/perception_mae_m"] = float((image_error * mask).sum() / denom)
                if mode == "intact":
                    copy_error = (depth - target_image).abs().flatten(1).mean(-1) * span
                    out[f"wm_diag/copy/h{j+1}/perception_mae_m"] = float((copy_error * mask).sum() / denom)
        reconstructed = runner.latent_dec(z0)
        # Both observations are in [0,1]. Height's span is 2 m; depth's span is 2.5 m.
        out["wm_diag/reconstruction_mae_m"] = float((reconstructed - depth).abs().mean() * span)
        out["wm_diag/mean_image_mae_m"] = float((depth.mean(0, keepdim=True) - depth).abs().mean() * span)
        out["wm_diag/latent_std"] = float(z0.std(0).mean())
        return out
