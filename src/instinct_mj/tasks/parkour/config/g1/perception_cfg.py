"""No-reference velocity locomotion: depth versus ideal local elevation maps.

Grid convention follows legged_gym / Isaac Lab: yaw aligned 1.6 x 1.0 m,
0.1 m spacing, root-height minus ground-height minus 0.5 m, clipped to [-1,1].
This map is simulation privileged information, not a reconstructed camera map.
"""

from mjlab.envs.mdp import height_scan
from mjlab.managers import ObservationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from .g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_final_cfg
from .rl_cfg import g1_parkour_supermimic_instinct_rl_cfg


def elevation_image(env):
    """Current pre-reset sensor sample, [B,11,17], y-major; [0,1] storage."""
    height = height_scan(env, sensor_name="perception_height", offset=0.5, miss_value=1.0)
    return ((height.clamp(-1.0, 1.0) + 1.0) * 0.5).reshape(-1, 11, 17)


def perception_env_cfg(modality: str, play: bool = False):
    assert modality in ("depth", "height")
    # Use identical training termination semantics in evaluation/play as well.
    cfg = instinct_g1_parkour_amp_final_cfg(play=False, shoe=True)
    cfg.rewards["rewards"]["dont_wait"].params.update(
        velocity_temperature=0.05, command_min=0.1, command_full=0.3
    )
    # Pure velocity commands, sampled independently of target positions on every terrain.
    command = cfg.commands["base_velocity"]
    command.random_velocity_terrain = list(cfg.scene.terrain.terrain_generator.sub_terrains)
    command.ranges.lin_vel_x = (0.2, 1.0)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    command.rel_standing_envs = 0.0
    cfg.rewards["rewards"]["heading_error"].weight = 0.0
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "motion_reference")
    cfg.observations.pop("amp_policy")
    cfg.observations.pop("amp_reference")
    cfg.terminations.pop("dataset_exhausted")
    # Match the two temporal windows: eight consecutive observations, no extra delay.
    for group in ("policy", "critic"):
        cfg.observations[group].terms["depth_image"].params.update(
            history_skip_frames=1, delayed_frame_ranges=(0, 0)
        )
    if modality == "height":
        cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "camera") + (
            RayCastSensorCfg(
                name="perception_height", frame=ObjRef(type="body", name="pelvis", entity="robot"),
                pattern=GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
                ray_alignment="yaw", max_distance=10.0, include_geom_groups=(0,), debug_vis=False,
            ),
        )
        for group in ("policy", "critic"):
            terms = cfg.observations[group].terms
            terms.pop("depth_image")
            terms["height_image"] = ObservationTermCfg(
                func=elevation_image, history_length=8, flatten_history_dim=False,
            )
    return cfg


def perception_agent_cfg(modality: str):
    cfg = g1_parkour_supermimic_instinct_rl_cfg()
    cfg.style_reward_coef, cfg.style_minibatches = 0.0, 0
    cfg.wm_use_privileged_scan = False
    cfg.wm_diagnostics_interval = 1000
    cfg.fail_head, cfg.wm_failure_frac, cfg.wm_holdout_envs = True, 0.25, 0.1
    cfg.max_iterations = 20000
    cfg.save_interval = 500
    if modality == "height":
        for enc in (cfg.policy.encoder_configs, cfg.policy.critic_encoder_configs):
            enc.depth_encoder.component_names = ["height_image"]
    return cfg
