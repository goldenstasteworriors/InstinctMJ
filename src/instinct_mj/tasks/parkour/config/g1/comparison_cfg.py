"""PPO controls for the existing SuperMimic environments, without changing their MDPs."""

from .rl_cfg import g1_parkour_amp_instinct_rl_cfg


def comparison_ppo_cfg(no_amp=False):
    cfg = g1_parkour_amp_instinct_rl_cfg()
    cfg.load_run = None
    cfg.resume = False
    cfg.save_interval = 500
    if no_amp:
        # PPO accepts the shared dataclass's optional discriminator fields as kwargs,
        # but does not construct a discriminator or consume reference observations.
        cfg.algorithm.class_name = "PPO"
    return cfg
