"""Native parkour configuration; keep the existing reward and termination managers."""

from __future__ import annotations

from pathlib import Path

from mjlab.sensor import ContactMatch, ContactSensorCfg

from instinct_mj.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import instinct_g1_parkour_amp_env_cfg


def make_config(data: str, envs: int, rows: int, cols: int, seed: int):
  cfg = instinct_g1_parkour_amp_env_cfg()
  cfg.seed = seed
  cfg.scene.num_envs = envs
  cfg.scene.terrain.terrain_generator.num_rows = rows
  cfg.scene.terrain.terrain_generator.num_cols = cols
  cfg.scene.terrain.max_init_terrain_level = min(5, rows - 1)
  cfg.scene.terrain.terrain_generator.seed = seed
  cfg.scene.terrain.terrain_generator.use_cache = True
  sensors = list(cfg.scene.sensors)
  for sensor in sensors:
    if sensor.name == "motion_reference":
      for motion in sensor.motion_buffers.values():
        motion.path = str(Path(data).resolve())
        motion.filtered_motion_selection_filepath = str(Path(data).resolve() / "selection.yaml")
  sensors.append(ContactSensorCfg(name="supermimic_contacts",
    primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
    reduce="netforce", history_length=1))
  cfg.scene.sensors = tuple(sensors)
  return cfg
