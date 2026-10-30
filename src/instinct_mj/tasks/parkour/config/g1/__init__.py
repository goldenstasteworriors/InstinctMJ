"""Register Instinct Mj parkour G1 tasks."""

# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from instinct_mj.tasks.registry import register_instinct_task


def _parkour_amp_env_cfg(play: bool, shoe: bool = True, smooth_dont_wait: bool = False):
    from .g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg

    cfg = instinct_g1_parkour_amp_final_cfg(play=play, shoe=shoe)
    if smooth_dont_wait:
        cfg.rewards["rewards"]["dont_wait"].params.update(
            velocity_temperature=0.05, command_min=0.1, command_full=0.3
        )
    return cfg


def g1_parkour_amp_instinct_rl_cfg():
    from .rl_cfg import g1_parkour_amp_instinct_rl_cfg as _cfg

    return _cfg()


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_instinct_rl_cfg,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-G1-Play-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True),
    instinct_rl_cfg_factory=g1_parkour_amp_instinct_rl_cfg,
)


# SuperMimic uses a smooth dont_wait reward in both the env and imagination.
PARKOUR_SUPERMIMIC_RUNNER_ENTRY_POINT = "instinct_mj.rl.supermimic.parkour.runner:ParkourSuperMimicRunner"


def _perception_env(modality, play=False):
    from .perception_cfg import perception_env_cfg
    return perception_env_cfg(modality, play)


def _perception_agent(modality):
    from .perception_cfg import perception_agent_cfg
    return perception_agent_cfg(modality)


for _modality in ("depth", "height"):
    register_instinct_task(
        task_id=f"Instinct-Velocity-NoAmp-{_modality.title()}-SuperMimic-G1-v0",
        env_cfg_factory=lambda modality=_modality: _perception_env(modality),
        play_env_cfg_factory=lambda modality=_modality: _perception_env(modality, True),
        instinct_rl_cfg_factory=lambda modality=_modality: _perception_agent(modality),
        runner_entry_point=PARKOUR_SUPERMIMIC_RUNNER_ENTRY_POINT,
    )


def g1_parkour_supermimic_instinct_rl_cfg():
    from .rl_cfg import g1_parkour_supermimic_instinct_rl_cfg as _cfg

    return _cfg()


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-SuperMimic-G1-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False, shoe=True, smooth_dont_wait=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True, smooth_dont_wait=True),
    instinct_rl_cfg_factory=g1_parkour_supermimic_instinct_rl_cfg,
    runner_entry_point=PARKOUR_SUPERMIMIC_RUNNER_ENTRY_POINT,
)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-SuperMimic-G1-Play-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True, smooth_dont_wait=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=True, shoe=True, smooth_dont_wait=True),
    instinct_rl_cfg_factory=g1_parkour_supermimic_instinct_rl_cfg,
    runner_entry_point=PARKOUR_SUPERMIMIC_RUNNER_ENTRY_POINT,
)


def _comparison_ppo(no_amp=False):
    from .comparison_cfg import comparison_ppo_cfg
    return comparison_ppo_cfg(no_amp)


register_instinct_task(
    task_id="Instinct-Parkour-Target-Amp-MatchedPPO-G1-v0",
    env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False, shoe=True, smooth_dont_wait=True),
    play_env_cfg_factory=lambda: _parkour_amp_env_cfg(play=False, shoe=True, smooth_dont_wait=True),
    instinct_rl_cfg_factory=_comparison_ppo,
)

register_instinct_task(
    task_id="Instinct-Velocity-NoAmp-Depth-MatchedPPO-G1-v0",
    env_cfg_factory=lambda: _perception_env("depth"),
    play_env_cfg_factory=lambda: _perception_env("depth"),
    instinct_rl_cfg_factory=lambda: _comparison_ppo(True),
)
