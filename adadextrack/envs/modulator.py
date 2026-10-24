"""Hierarchical control with the original Planner2k5 modulation and step order.

This adapter wraps the unchanged 22-action physics task. The PPO agent sees 202
actions, while the robot and reward continue to receive 22 tracker actions.
Reference/latent modulation enters the *next* cached tracker observation; the
current residual acts immediately, matching the original task's execution order.
"""

import numpy as np
import torch
from gym.spaces import Box
from adadextrack.control import (
    MODULATOR_ACTIONS, cache_tracker_observation, combine_actions, modulate_observation,
)


class ModulatorEnv:
    def __init__(self, env, tracker):
        self.env, self.tracker = env, tracker.requires_grad_(False).eval()
        self.action_space = Box(-1., 1., shape=(202,), dtype=np.float32)
        self.observation_space = env.observation_space
        self.num_envs = env.num_envs
        self.num_actions = self.num_acts = 202
        self.tracker_observation = torch.zeros((env.num_envs, 273), device=env.device)
        self.planner_action_tensor = torch.zeros((env.num_envs, 202), device=env.device)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        observation = self.env.reset()
        # reset() is only the startup observation; physical reset occurs in step.
        self.tracker_observation.copy_(observation["obs"])
        return observation

    @torch.no_grad()
    def step(self, actions):
        self.planner_action_tensor.copy_(actions.clamp(-1., 1.))
        self.tracker.eval()
        with torch.no_grad():
            result = self.tracker(dict(is_train=False, prev_actions=None,
                                       obs=self.tracker_observation, rnn_states=None))
            control = combine_actions(result["mus"], self.planner_action_tensor)
        observation, reward, done, info = self.env.step(control)
        # Use raw task observations so clipping occurs after modulation, exactly
        # like the original VecTask's obs_tracker return path.
        self.tracker_observation = cache_tracker_observation(
            self.env.obs_buf, self.planner_action_tensor,
            self.env.shadow_hand_dof_lower_limits, self.env.shadow_hand_dof_upper_limits,
            self.env.clip_obs,
        )
        return observation, reward, done, info

    def zero_actions(self):
        return torch.zeros((self.num_envs, 202), device=self.env.device)
