"""Reads the SuperMimic data of every env from the live mjlab parkour env (``Instinct-Parkour-Target-Amp-G1-*``).

Layouts (shared with ``mdp.py``):

- state ``S = (p, R, v, w, q, qd)`` of the floating base (``pelvis``), world frame, from ``qpos`` / ``qvel``.
- contacts (``NUM_CONTACTS``): ``[left foot, right foot, torso, undesired bodies...]``, each "max force norm over the sensor
  history > threshold" of the task's ``contact_forces`` (feet), ``torso_contact_forces`` and ``undesired_contact_forces``
  sensors (the indicators of the ``feet_*``, ``base_contact`` and ``undesired_contacts`` terms).
- feet timers (4): ``[air L, air R, contact L, contact R]`` = ``current_air_time`` / ``current_contact_time`` of the feet.
- command (``CMD_DIM``): ``[target_w (3), max_command_b (3), is_standing (1), is_random_velocity (1), random velocity
  (3: x, y, yaw), env_origin (3)]`` of the ``base_velocity`` ``PoseVelocityCommand`` (its parameters; the velocity command
  itself is a function of the state, see ``ParkourMDP.velocity_command``).
"""

from __future__ import annotations

import torch

from ..rotations import quat_to_mat

CMD_DIM = 14
CMD_TARGET, CMD_MAX, CMD_STANDING, CMD_RANDOM, CMD_RANDOM_VEL, CMD_ORIGIN = (
    slice(0, 3), slice(3, 6), slice(6, 7), slice(7, 8), slice(8, 11), slice(11, 14)
)  # fmt: skip
CONTACT_FEET, CONTACT_TORSO = slice(0, 2), 2  # then the undesired-contact bodies


class ParkourSimReader:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        idx = self.robot.indexing
        self.free_q, self.free_v = idx.free_joint_q_adr.long(), idx.free_joint_v_adr.long()
        self.joint_q, self.joint_v = idx.joint_q_adr.long(), idx.joint_v_adr.long()
        self.feet_sensor = env.scene["contact_forces"]
        self.torso_sensor = env.scene["torso_contact_forces"]
        self.undesired_sensor = env.scene["undesired_contact_forces"]
        terms = env.reward_manager.get_term_cfg
        self.feet_threshold = float(terms("feet_slide").params["threshold"])
        self.undesired_threshold = float(terms("undesired_contacts").params["threshold"])
        self.torso_threshold = float(env.termination_manager.get_term_cfg("base_contact").params["threshold"])
        self.num_contacts = 3 + self.undesired_sensor.data.force_history.shape[1]
        self.command = env.command_manager.get_term("base_velocity")
        self.delayed = [a for a in self.robot.actuators if getattr(a, "_delay_buffer", None) is not None]

    def state(self):
        """Exact simulator state (from ``qpos`` / ``qvel``, no derived-quantity staleness)."""
        data = self.env.sim.data
        qpos, qvel = data.qpos, data.qvel
        p = qpos[:, self.free_q[:3]].clone()
        R = quat_to_mat(qpos[:, self.free_q[3:7]])
        v = qvel[:, self.free_v[:3]].clone()  # linear velocity of the body-frame origin, world frame
        w = (R @ qvel[:, self.free_v[3:6]].unsqueeze(-1)).squeeze(-1)  # free-joint angular velocity is in the body frame
        return p, R, v, w, qpos[:, self.joint_q].clone(), qvel[:, self.joint_v].clone()

    def last_action(self) -> torch.Tensor:
        return self.env.action_manager.action.clone()

    @staticmethod
    def _indicator(sensor, threshold: float) -> torch.Tensor:
        return (torch.linalg.vector_norm(sensor.data.force_history, dim=-1).max(-1).values > threshold).float()

    def contact(self) -> torch.Tensor:
        """(N, NUM_CONTACTS) contact indicators, see the module docstring."""
        return torch.cat(
            (
                self._indicator(self.feet_sensor, self.feet_threshold),
                self._indicator(self.torso_sensor, self.torso_threshold),
                self._indicator(self.undesired_sensor, self.undesired_threshold),
            ),
            -1,
        )

    def feet_timers(self) -> torch.Tensor:
        d = self.feet_sensor.data
        return torch.cat((d.current_air_time, d.current_contact_time), -1).clone()

    def command_params(self) -> torch.Tensor:
        """(N, CMD_DIM) parameters of the velocity command, see the module docstring."""
        c = self.command
        random_vel = torch.stack((c.random_lin_vel_x, c.random_lin_vel_y, c.random_ang_vel_z), -1)
        return torch.cat(
            (
                c.pos_command_w,
                c.max_command_b,
                c.is_standing_env.float().unsqueeze(-1),
                c.random_velocity_indices.float().unsqueeze(-1),
                random_vel,
                self.env.scene.env_origins,
            ),
            -1,
        ).clone()

    def command_time_left(self) -> torch.Tensor:
        """(N,) time [s] until the command (target) is resampled."""
        return self.command.time_left.clone()

    def actuator_lags(self) -> torch.Tensor:
        """(N, num_joints) actuation delay [physics steps] of every joint, sampled per episode by the delayed actuators."""
        lags = torch.zeros(self.env.num_envs, self.robot.num_joints, device=self.env.device)
        for act in self.delayed:
            lags[:, act.target_ids.long()] = act._delay_buffer.current_lags.float().unsqueeze(-1)
        return lags

    def friction(self) -> torch.Tensor:
        """(N, 1) friction coefficient of the robot geoms (startup randomization, the same for all geoms of an env)."""
        geom_ids = self.robot.indexing.geom_ids.long()
        return self.env.sim.model.geom_friction[:, geom_ids, 0].mean(-1, keepdim=True)
