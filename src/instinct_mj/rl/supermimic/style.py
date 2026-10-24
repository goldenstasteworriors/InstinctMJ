"""Adversarial style reward (AMP / WASABI) for SuperMimic: a discriminator on windows of the robot state, whose reward is
differentiable w.r.t. the (imagined) states, so it is maximized by the BPTT of the policy like the task reward.

The discriminator, its losses and its reward follow ``instinct_rl.algorithms.WasabiAlgoMixin`` (the network is the same
``instinct_rl.modules.Discriminator``): it is trained only on REAL data, windows of the policy's own states (the env's
``amp_policy`` observation, or the same window rebuilt from buffered states) against windows of the reference motions (the
env's ``amp_reference`` observation). Task agnostic: the task only provides the windows.
"""

from __future__ import annotations

import instinct_rl.modules as instinct_modules
import torch
from torch import optim


class StyleReward:
    def __init__(
        self,
        input_segment: dict[str, tuple],
        device: str,
        discriminator_kwargs: dict,
        optimizer_class_name: str = "AdamW",
        optimizer_kwargs: dict | None = None,
        reward_coef: float = 1.0,
        reward_type: str = "quad",
        loss_func: str = "MSELoss",
        loss_coef: float = 1.0,
        gradient_penalty_coef: float = 10.0,
        weight_decay_coef: float = 0.0,
        logit_weight_decay_coef: float = 0.0,
        gradient_tolerance: float = 0.0,
    ):
        """Arguments as the ``discriminator_*`` arguments of ``WasabiAlgoMixin`` (``input_segment``: the format of the
        ``amp_policy`` group)."""
        self.discriminator = instinct_modules.Discriminator(input_segment=input_segment, **discriminator_kwargs).to(device)
        self.optimizer = getattr(optim, optimizer_class_name)(self.discriminator.parameters(), **(optimizer_kwargs or {}))
        self.reward_coef, self.reward_type, self.loss_func = reward_coef, reward_type, loss_func
        self.loss_coef, self.gradient_penalty_coef = loss_coef, gradient_penalty_coef
        self.weight_decay_coef, self.logit_weight_decay_coef = weight_decay_coef, logit_weight_decay_coef
        self.gradient_tolerance = gradient_tolerance

    def reward(self, window: torch.Tensor) -> torch.Tensor:
        """Style reward (B,) of the windows (B, window_dim), times ``reward_coef``; differentiable w.r.t. the windows (the
        discriminator parameters are not trained by it)."""
        disc = self.discriminator(window).squeeze(-1)
        if self.reward_type == "log":
            reward = -torch.log(1 - torch.clamp(torch.sigmoid(disc), 1e-6, 1 - 1e-6))
        elif self.reward_type == "quad":
            reward = torch.clamp(1 - 0.25 * torch.square(disc - 1), min=0)
        elif self.reward_type == "wasserstein":
            reward = disc
        else:
            raise NotImplementedError(f"discriminator reward type {self.reward_type}")
        return self.reward_coef * reward

    def update(self, policy_windows: torch.Tensor, reference_windows: torch.Tensor) -> dict[str, torch.Tensor]:
        """One discriminator step on a minibatch of policy and reference windows (the losses of
        ``WasabiAlgoMixin.compute_amp_losses``). Returns the losses and statistics (detached)."""
        policy_d = self.discriminator(policy_windows)
        reference_d = self.discriminator(reference_windows)
        if self.loss_func == "WassersteinLoss":
            policy_loss, reference_loss = policy_d.mean(), -reference_d.mean()
        elif self.loss_func == "BCEWithLogitsLoss":
            policy_loss = torch.nn.functional.binary_cross_entropy_with_logits(policy_d, torch.zeros_like(policy_d))
            reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(reference_d, torch.ones_like(reference_d))
        elif self.loss_func == "MSELoss":
            policy_loss = torch.nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            reference_loss = torch.nn.functional.mse_loss(reference_d, torch.ones_like(reference_d))
        else:
            raise NotImplementedError(f"discriminator loss function {self.loss_func}")
        discriminator_loss = 0.5 * (policy_loss + reference_loss)
        # gradient penalty w.r.t. the inputs (policy and reference windows), as ``compute_discriminator_gradient``
        gradient_penalty = torch.zeros((), device=policy_windows.device)
        if self.gradient_penalty_coef > 0:
            combined = torch.cat((policy_windows, reference_windows), 0).clone().requires_grad_(True)
            disc = self.discriminator(combined)
            grad = torch.autograd.grad(disc, combined, torch.ones_like(disc), create_graph=True, retain_graph=True)[0]
            gradient_penalty = torch.clamp(grad.norm(2, dim=1) - self.gradient_tolerance, 0).pow(2).mean()
        weight_decay = sum(torch.sum(p**2) for p in self.discriminator.parameters())
        logit_weight_decay = torch.zeros((), device=policy_windows.device)
        if self.logit_weight_decay_coef > 0:
            logit_weight_decay = torch.sum(self.discriminator.logit_layer_weights() ** 2)
        # NOTE: the loss weights are those of ``WasabiAlgoMixin.update`` (``getattr(self, name + "_coef", 1.0)``): the
        # logit weight decay term is looked up as ``logit_weight_decay_coef``, which WASABI does not define, so it enters
        # with weight 1.0 there (its ``discriminator_logit_weight_decay_coef`` only switches it on). Kept for parity.
        loss = (
            self.loss_coef * discriminator_loss
            + self.gradient_penalty_coef * gradient_penalty
            + self.weight_decay_coef * weight_decay
            + 1.0 * logit_weight_decay
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {
            "discriminator_loss": discriminator_loss.detach(), "gradient_penalty": gradient_penalty.detach(),
            "discriminator_policy": policy_d.detach().mean(), "discriminator_reference": reference_d.detach().mean(),
        }  # fmt: skip

    def state_dict(self) -> dict:
        return {"discriminator": self.discriminator.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
