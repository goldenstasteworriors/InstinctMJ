"""Motion-terrain paired Perceptive Shadowing for SuperMimic training."""

from pathlib import Path

from instinct_mj.tasks.shadowing.perceptive.config.g1.perceptive_shadowing_cfg import G1PerceptiveShadowingEnvCfg


def make_tracking_config(data: str, envs: int, seed: int, evaluation: bool = False):
  cfg = G1PerceptiveShadowingEnvCfg()
  cfg.seed = seed
  cfg.scene.num_envs = envs
  cfg.monitors = {}
  root = str(Path(data).resolve())
  terrain = cfg.scene.terrain.terrain_generator
  # Exactly one difficulty stratum for every mesh in the six-terrain dataset.
  import yaml
  metadata = yaml.safe_load(Path(root, "metadata.yaml").read_text())
  terrain.num_rows = len(metadata["terrains"])
  terrain.num_cols = 1
  terrain.curriculum = True
  terrain.seed = seed
  mesh = terrain.sub_terrains["motion_matched"]
  mesh.path, mesh.metadata_yaml = root, str(Path(root, "metadata.yaml"))
  for sensor in cfg.scene.sensors:
    sensor.debug_vis = False
    if sensor.name == "motion_reference":
      # 1.7 seconds covers a 0.64 second imagination plus the actor's 0.9 second goal.
      sensor.num_frames = 18
      sensor.frame_interval_s = 0.1
      for motion in sensor.motion_buffers.values():
        motion.path, motion.metadata_yaml = root, mesh.metadata_yaml
        if evaluation:
          motion.motion_bin_length_s = None
          motion.motion_start_from_middle_range = [0.0, 0.0]
          motion.env_starting_stub_sampling_strategy = "independent"
  if evaluation:
    cfg.curriculum = {}
    cfg.events.pop("bin_fail_counter_smoothing", None)
    cfg.events["reset_robot"].params["randomize_pose_range"] = {}
    cfg.events["reset_robot"].params["randomize_joint_pos_range"] = (0.0, 0.0)
    cfg.observations["policy"].enable_corruption = False
  return cfg
