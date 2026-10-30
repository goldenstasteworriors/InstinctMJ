"""Runner config of SuperMimic for the parkour task: the SuperMimic hyperparameters plus the adversarial style reward."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SuperMimicRunnerCfg


@dataclass(kw_only=True)
class ParkourSuperMimicRunnerCfg(SuperMimicRunnerCfg):
    """Hyperparameters of :class:`ParkourSuperMimicRunner`, mixed into the task's agent cfg (flat ``--agent.<field>``).
    The ``style_*`` fields are the ``discriminator_*`` arguments of the task's PPO ``WasabiPPO`` config."""

    buffer_steps: int = 800
    wm_use_privileged_scan: bool = True
    """Legacy direct terrain input; disable to isolate the learned perception latent."""
    wm_diagnostics_interval: int = 0
    perception_geometry_weight: float = 0.0
    """Train-only local terrain supervision of the visual encoder; no extra policy/WM input."""
    critic_real_weight: float = 0.0
    """Mix real one-step TD loss into critic fitting (0 preserves imagined-only fitting).

    Real targets use task + current AMP reward and bootstrap through truncations,
    but not failures. Actor optimization and world-model training are unchanged.
    """
    """Held-out physical-state prediction tests with intact/shuffled/mean perception."""
    """Replay buffer length in vectorized env steps (default 800; the depth stacks dominate the memory)."""
    keep_latest_checkpoints: int = 2
    """Retain this many recent policy/sidecar pairs plus the best training-score pair; 0 disables pruning."""
    command_margin_steps: int = 2
    """Imagination starts need the command (target) to stay for ``horizon + command_margin_steps`` more steps (default 2)."""

    # ---- adversarial style reward (AMP / WASABI) ----
    style_reward_coef: float = 0.25
    """Weight of the style reward added to the per-step task reward (``discriminator_reward_coef``, default 0.25)."""
    style_reward_type: str = "quad"
    """``discriminator_reward_type`` (default "quad")."""
    style_loss_func: str = "MSELoss"
    """``discriminator_loss_func`` (default "MSELoss")."""
    style_gradient_penalty_coef: float = 5.0
    """``discriminator_gradient_penalty_coef`` (default 5.0)."""
    style_weight_decay_coef: float = 3e-4
    """``discriminator_weight_decay_coef`` (default 3e-4)."""
    style_logit_weight_decay_coef: float = 0.04
    """``discriminator_logit_weight_decay_coef`` (default 0.04)."""
    style_discriminator_kwargs: dict = field(default_factory=lambda: {"hidden_sizes": [1024, 512], "nonlinearity": "ReLU"})
    """``discriminator_kwargs`` (default MLP [1024, 512] ReLU)."""
    style_optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1.0e-4, "betas": [0.9, 0.999]})
    """AdamW kwargs of the discriminator (``discriminator_optimizer_kwargs``)."""
    style_batch: int = 2048
    """Policy / reference windows per discriminator minibatch (default 2048)."""
    style_minibatches: int = 2
    """Discriminator minibatches per iteration (default 2)."""
    style_recent_steps: int = 48
    """Policy windows are drawn from the most recent ``style_recent_steps`` buffered steps (near on-policy, default 48)."""
    style_reference_steps: int = 100
    """Length of the FIFO of the env's ``amp_reference`` windows (vectorized steps, default 100)."""
