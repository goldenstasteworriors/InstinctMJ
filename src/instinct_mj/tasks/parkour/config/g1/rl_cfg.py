"""Instinct-RL configuration for the G1 parkour AMP task."""

from __future__ import annotations

import os

from dataclasses import dataclass, field

from instinct_mj.rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mj.rl.supermimic.parkour.config import ParkourSuperMimicRunnerCfg


@dataclass(kw_only=True)
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size: int = 128
    channels: list = field(default_factory=lambda: [4])
    kernel_sizes: list = field(default_factory=lambda: [3])
    strides: list = field(default_factory=lambda: [1])
    hidden_sizes: list = field(default_factory=lambda: [256, 256])
    paddings: list = field(default_factory=lambda: [1])
    nonlinearity: str = "ReLU"
    use_maxpool: bool = True
    component_names: list = field(
        default_factory=lambda: [
            "depth_image",
        ]
    )


@dataclass(kw_only=True)
class EncoderConfigs:
    depth_encoder: object = field(default_factory=lambda: DepthEncoderConv2dCfg())


@dataclass(kw_only=True)
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std: float = 1.0
    num_moe_experts: int = 4
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    encoder_configs: object = field(default_factory=lambda: EncoderConfigs())
    critic_encoder_configs: object = field(default_factory=lambda: EncoderConfigs())


@dataclass(kw_only=True)
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name: str = "WasabiPPO"
    discriminator_kwargs: dict = field(
        default_factory=lambda: {
            "hidden_sizes": [1024, 512],
            "nonlinearity": "ReLU",
        }
    )
    discriminator_reward_coef: float = 0.25
    discriminator_reward_type: str = "quad"
    discriminator_loss_func: str = "MSELoss"
    discriminator_gradient_penalty_coef: float = 5.0
    discriminator_optimizer_class_name: str = "AdamW"
    discriminator_weight_decay_coef: float = 3e-4
    discriminator_logit_weight_decay_coef: float = 0.04
    discriminator_optimizer_kwargs: dict = field(
        default_factory=lambda: {
            "lr": 1.0e-4,
            "betas": [0.9, 0.999],
        }
    )
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.006
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(kw_only=True)
class G1ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    policy_observation_group: str = "policy"
    critic_observation_group: str = "critic"
    max_iterations: int = 30000
    save_interval: int = 5000
    experiment_name: str = "g1_parkour"
    resume: bool = False
    load_run: str = "^(?!_play$).*"
    empirical_normalization: bool = False
    policy: object = field(default_factory=lambda: MoEPolicyCfg())
    algorithm: object = field(default_factory=lambda: AmpAlgoCfg())


def g1_parkour_amp_instinct_rl_cfg() -> G1ParkourPPORunnerCfg:
    return G1ParkourPPORunnerCfg()


@dataclass(kw_only=True)
class G1ParkourSuperMimicRunnerCfg(InstinctRlOnPolicyRunnerCfg, ParkourSuperMimicRunnerCfg):
    """Agent cfg of the parkour task trained with SuperMimic (``ParkourSuperMimicRunner``): the task's policy (the same
    MoE actor and depth encoder as the PPO task) plus the SuperMimic and style-reward hyperparameters (flat,
    ``--agent.<field>``). There is no PPO algorithm and no observation normalization (as in the PPO task)."""

    policy: object = field(default_factory=lambda: MoEPolicyCfg())
    algorithm: None = None
    num_steps_per_env: None = None
    max_iterations: int = 320_000
    save_interval: int = 2000
    log_interval: int = 50
    experiment_name: str = "g1_parkour_supermimic"
    load_run: str | None = None

    def __post_init__(self):
        self.resume = self.load_run is not None
        self.run_name = "supermimic" + (f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "")


def g1_parkour_supermimic_instinct_rl_cfg() -> G1ParkourSuperMimicRunnerCfg:
    return G1ParkourSuperMimicRunnerCfg()
