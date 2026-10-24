"""Policy arithmetic shared by simulation and hardware interface examples.

This module depends only on PyTorch. It does not import a simulator or a driver.
"""

import torch


OBSERVATIONS = 273
TRACKER_ACTIONS = 22
MODULATOR_ACTIONS = 202
REFERENCE_OFFSETS = (0, 1, 2, 4, 12)


def reference_indices(frame, lengths):
    """Select current/lookahead frames, repeating the last available frame."""
    offsets = frame.new_tensor(REFERENCE_OFFSETS)
    return torch.minimum(frame.unsqueeze(-1) + offsets, (lengths - 1).unsqueeze(-1))


def relative_joint_increment(actions, speed_scale, dt, control_frequency_inv):
    """Convert normalized actions into joint-position increments in radians."""
    return speed_scale * dt * actions * 2.0 * control_frequency_inv


def modulate_observation(observation, action, lower, upper):
    """Adjust raw tracker features before observation clipping and RMS."""
    if observation.shape[-1] != OBSERVATIONS or action.shape != observation.shape[:-1] + (MODULATOR_ACTIONS,):
        raise ValueError("Expected 273 observations and 202 modulator actions")
    result = observation.clone()
    # Four object poses (xyz + xyzw); the learned quaternion offsets are additive.
    result[..., 71:99] += .1 * .10 * action[..., :28]
    adjustment = .1 * .314 * action[..., 28:116].reshape(*action.shape[:-1], 4, 22)
    joint_refs = result[..., 121:209].reshape(*action.shape[:-1], 4, 22)
    joint_refs += 2.0 * adjustment / (upper - lower)
    result[..., 209:273] = .6 * action[..., 116:180]
    return result


def combine_actions(tracker_mean, modulator_action):
    """Add the current residual to the unbounded BC mean, then clip once."""
    if tracker_mean.shape[-1] != TRACKER_ACTIONS or modulator_action.shape != tracker_mean.shape[:-1] + (MODULATOR_ACTIONS,):
        raise ValueError("Expected matching 22-action BC and 202-action modulator batches")
    return (tracker_mean + modulator_action[..., 180:202]).clamp(-1., 1.)


def cache_tracker_observation(raw_observation, action, lower, upper, clip_obs):
    """Prepare the next cycle's tracker observation using this cycle's action."""
    return modulate_observation(raw_observation, action, lower, upper).clamp(-clip_obs, clip_obs)
