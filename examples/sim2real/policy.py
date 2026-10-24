"""Observation and policy examples for a single xArm6 / LEAP Hand system."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

from adadextrack.control import (
    cache_tracker_observation, combine_actions, reference_indices, relative_joint_increment,
)
from adadextrack.data.dataset import read_manifest, load_task_reference_tensors
from adadextrack.data.latents import DEFAULT_LATENTS, check_latent_metadata
from adadextrack.inference import ROOT, build_model


POLICY_JOINT_NAMES = tuple("joint{}".format(i) for i in range(1, 7)) + tuple(
    "joint_{}.0".format(i) for i in (1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11)
)
POSE_LINKS = ("palm_lower", "index_tip_head", "ring_tip_head", "middle_tip_head", "thumb_tip_head")
POLICY_HZ = 12


def finite_tensor(value, shape, device):
    result = torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.shape != shape or not torch.isfinite(result).all():
        raise ValueError("Expected finite tensor with shape {}".format(shape))
    return result


class RobotKinematics:
    """Map named policy joints to the released URDF for limits and FK."""

    def __init__(self, urdf_path, device="cpu"):
        import pytorch_kinematics as pk

        self.pk = pk
        source = Path(urdf_path).read_bytes()
        self.device = device
        self.chain = pk.build_chain_from_urdf(source).to(dtype=torch.float32, device=device)
        names = self.chain.get_joint_parameter_names()
        if len(names) != 22 or set(names) != set(POLICY_JOINT_NAMES):
            raise ValueError("Expected the released xArm6 / LEAP joint names")
        self.fk_order = torch.tensor([POLICY_JOINT_NAMES.index(name) for name in names], device=device)
        joints = {joint.attrib["name"]: joint for joint in ET.fromstring(source).findall("joint")}
        self.lower = torch.tensor([float(joints[name].find("limit").attrib["lower"])
                                   for name in POLICY_JOINT_NAMES], device=device)
        self.upper = torch.tensor([float(joints[name].find("limit").attrib["upper"])
                                   for name in POLICY_JOINT_NAMES], device=device)

    @torch.no_grad()
    def poses(self, qpos):
        qpos = finite_tensor(qpos, (22,), self.device)
        transforms = self.chain.forward_kinematics(qpos[self.fk_order])
        poses = []
        for name in POSE_LINKS:
            matrix = transforms[name].get_matrix()[0]
            wxyz = self.pk.matrix_to_quaternion(matrix[:3, :3].unsqueeze(0))[0]
            poses.append(torch.cat((matrix[:3, 3], wxyz[1:], wxyz[:1])))
        return torch.stack(poses)


def build_observation(frame, qpos, link_poses, object_pose, reference, latent, lower, upper):
    """Assemble raw 273 features in metres, radians and xyzw quaternion order."""
    device = latent.device
    qpos = finite_tensor(qpos, (22,), device)
    links = finite_tensor(link_poses, (5, 7), device)
    obj = finite_tensor(object_pose, (7,), device)
    latent = finite_tensor(latent, (64,), device)
    length = reference["qpos"].shape[0]
    if not isinstance(frame, int) or not 0 <= frame < length:
        raise ValueError("Reference frame is outside the motion")
    for quats in (links[:, 3:7], obj[None, 3:7]):
        if not torch.allclose(torch.linalg.vector_norm(quats, dim=-1), torch.ones(quats.shape[0], device=device), atol=1e-4):
            raise ValueError("Measured poses must contain unit xyzw quaternions")
    indices = reference_indices(torch.tensor(frame, device=device), torch.tensor(length, device=device))
    goals = reference["object_pose"][indices]
    joints = reference["qpos"][indices]
    obs = torch.empty((1, 273), dtype=torch.float32, device=device)
    obs[0, :22] = (2.0 * qpos - upper - lower) / (upper - lower)
    obs[0, 22:57] = links.reshape(-1)
    obs[0, 57:64] = obj
    obs[0, 64:67] = goals[0, :3] - obj[:3]
    # q_object * conjugate(q_reference), in xyzw order.
    a, b = obj[3:7], goals[0, 3:7]
    obs[0, 67:70] = -a[3] * b[:3] + b[3] * a[:3] - torch.linalg.cross(a[:3], b[:3])
    obs[0, 70] = a[3] * b[3] + torch.dot(a[:3], b[:3])
    obs[0, 71:99] = goals[1:].reshape(-1)
    obs[0, 99:121] = qpos - joints[0]
    obs[0, 121:209] = ((2.0 * joints[1:] - upper - lower) / (upper - lower)).reshape(-1)
    obs[0, 209:273] = latent
    if not torch.isfinite(obs).all():
        raise ValueError("Non-finite policy observation")
    return obs


class TrackingPolicy:
    """Two-phase policy interface: act, then advance with the new raw observation."""

    def __init__(self, tracker, modulator, lower, upper, task_config):
        self.tracker = tracker.requires_grad_(False).eval()
        self.modulator = modulator.requires_grad_(False).eval() if modulator is not None else None
        self.lower, self.upper = lower, upper
        self.clip_obs = task_config["env"]["clipObservations"]
        self.speed = task_config["env"]["dofSpeedScale"]
        self.dt = task_config["sim"]["dt"]
        self.control_frequency_inv = task_config["env"]["controlFrequencyInv"]
        self.tracker_observation = None
        self.pending_action = None

    def _observation(self, obs):
        if obs.ndim != 2 or obs.shape[-1] != 273 or not torch.isfinite(obs).all():
            raise ValueError("Expected finite [batch, 273] observations")
        return obs

    @staticmethod
    def _mean(model, observation):
        # rl_games replaces the dictionary's obs entry during RMS normalization.
        return model(dict(is_train=False, prev_actions=None,
                          obs=observation.clone(), rnn_states=None))["mus"]

    def reset(self, observation):
        self.tracker_observation = self._observation(observation).clamp(-self.clip_obs, self.clip_obs).clone()
        self.pending_action = None

    @torch.no_grad()
    def act(self, observation):
        observation = self._observation(observation)
        if self.tracker_observation is None or self.pending_action is not None:
            raise RuntimeError("Call reset before act, and advance after each act")
        if observation.shape != self.tracker_observation.shape:
            raise ValueError("Observation batch changed without reset")
        if self.modulator is None:
            action = self._mean(self.tracker, observation.clamp(-self.clip_obs, self.clip_obs)).clamp(-1., 1.)
            self.pending_action = observation.new_zeros((observation.shape[0], 202))
        else:
            modulation = self._mean(self.modulator, observation.clamp(-self.clip_obs, self.clip_obs)).clamp(-1., 1.)
            self.pending_action = modulation.clone()
            action = combine_actions(self._mean(self.tracker, self.tracker_observation), modulation)
        if action.shape != (observation.shape[0], 22) or not torch.isfinite(action).all() or not torch.isfinite(self.pending_action).all():
            raise ValueError("Invalid policy output")
        return action

    def advance(self, raw_observation):
        observation = self._observation(raw_observation)
        if self.pending_action is None:
            raise RuntimeError("advance requires a preceding act")
        if observation.shape != self.tracker_observation.shape:
            raise ValueError("Observation batch changed without reset")
        if self.modulator is None:
            self.tracker_observation = observation.clamp(-self.clip_obs, self.clip_obs).clone()
        else:
            self.tracker_observation = cache_tracker_observation(
                observation, self.pending_action, self.lower, self.upper, self.clip_obs,
            )
        self.pending_action = None

    def joint_targets(self, previous, action):
        increment = relative_joint_increment(action, self.speed, self.dt, self.control_frequency_inv)
        return torch.maximum(torch.minimum(previous + increment, self.upper), self.lower)


@dataclass
class ExampleInputs:
    settings: dict
    reference: dict
    latent: torch.Tensor
    kinematics: RobotKinematics
    policy: TrackingPolicy

    def observe(self, frame, qpos, object_pose):
        return build_observation(frame, qpos, self.kinematics.poses(qpos), object_pose,
                                 self.reference, self.latent, self.kinematics.lower, self.kinematics.upper)


def load_example(config_path):
    """Read resources and frozen policies; no hardware or simulator is opened."""
    settings = yaml.safe_load(Path(config_path).read_text())
    device = settings["device"]
    entry, = read_manifest(ROOT / settings["reference_manifest"], [settings["reference_id"]], verify_hashes=True)
    pos, rot, joints, *_ = load_task_reference_tensors([entry], device)
    reference = dict(object_pose=torch.cat((pos[0], rot[0]), dim=-1), qpos=joints[0])
    tracker_path = ROOT / settings["tracker_checkpoint"]
    tracker = build_model(tracker_path, device)
    modulator_path = settings.get("modulator_checkpoint")
    modulator = build_model(ROOT / modulator_path, device, action_dim=202) if modulator_path else None
    for model in (tracker, modulator):
        if model is not None:
            check_latent_metadata(model.adadextrack_metadata, [entry.object_name], DEFAULT_LATENTS)
    if modulator is not None:
        from adadextrack.artifacts import sha256
        expected = (modulator.adadextrack_metadata or {}).get("tracker_sha256")
        if expected and sha256(tracker_path) != expected:
            raise ValueError("Modulator and BC tracker checkpoints are not a matching pair")
    latent_path, = [path for path in DEFAULT_LATENTS.glob("*_latent.npy")
                    if path.name[:-len("_latent.npy")].lower() == entry.object_name.lower()]
    latent = torch.as_tensor(np.load(str(latent_path), allow_pickle=False), device=device)
    kinematics = RobotKinematics(ROOT / settings["robot_urdf"], device)
    task = yaml.safe_load((ROOT / "configs/tracking.yaml").read_text())
    if abs(1.0 / (task["sim"]["dt"] * task["env"]["controlFrequencyInv"]) - POLICY_HZ) > .01:
        raise ValueError("Expected the 12 Hz tracking configuration")
    settings = dict(settings, object_name=entry.object_name)
    return ExampleInputs(settings, reference, latent, kinematics,
                         TrackingPolicy(tracker, modulator, kinematics.lower, kinematics.upper, task))
