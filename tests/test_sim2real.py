"""CPU checks of policy equivalence and injected hardware interfaces."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import gym
import numpy as np
import torch
import yaml

from adadextrack.control import combine_actions, reference_indices
from adadextrack.envs.modulator import ModulatorEnv
from examples.sim2real.interfaces import (
    ArmPD, HAND_ZERO, XArmVelocityController, hand_command, hand_state, object_in_base,
)
from examples.sim2real.policy import (
    ROOT, POSE_LINKS, POLICY_JOINT_NAMES, RobotKinematics, TrackingPolicy, build_observation,
)
from examples.sim2real.calibration import load_eye_to_hand, main as export_calibration


class BC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.full((22,), 1.5))
        self.observations = []

    def forward(self, data):
        self.observations.append(data["obs"].clone())
        result = data["obs"][:, 209:231] * .5 + self.bias
        data["obs"] = torch.zeros_like(data["obs"])
        return {"mus": result}


class Modulator(torch.nn.Module):
    def forward(self, data):
        obs = data["obs"]
        output = obs.new_zeros((obs.shape[0], 202))
        output[:, :180] = obs[:, :1] * .1 + 2.
        output[:, 180:] = -.5
        data["obs"] = torch.zeros_like(obs)
        return {"mus": output}


class TensorEnv:
    device = "cpu"
    num_envs = 1
    clip_obs = 5.
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(273,))
    shadow_hand_dof_lower_limits = torch.full((22,), -3.)
    shadow_hand_dof_upper_limits = torch.full((22,), 3.)

    def __init__(self):
        self.obs_buf = torch.zeros(1, 273)
        self.step_count = 0

    def reset(self):
        return {"obs": self.obs_buf.clamp(-5, 5)}

    def step(self, action):
        self.control = action.clone()
        self.step_count += 1
        self.obs_buf = torch.arange(273).reshape(1, -1).float() / 30 + self.step_count * .1
        return {"obs": self.obs_buf.clamp(-5, 5)}, 0., False, {}


class SDK:
    def __init__(self):
        self.commands = []
        self.mode = None
        self.state_code = self.feedback_code = self.command_code = 0

    def get_joint_states(self, is_radian):
        return self.feedback_code, [np.zeros(7), np.zeros(7), np.zeros(7)]

    def get_state(self):
        return self.state_code, 0

    def vc_set_joint_velocity(self, value, is_radian):
        self.commands.append(np.array(value))
        return self.command_code

    def motion_enable(self, enable):
        return 0

    def set_mode(self, mode):
        self.mode = mode
        return 0

    def set_state(self, state):
        return 0


class Sim2realTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.task = yaml.safe_load((ROOT / "configs/tracking.yaml").read_text())

    def test_multicycle_policy_matches_simulation_and_freezes_models(self):
        raw = TensorEnv()
        sim = ModulatorEnv(raw, BC())
        controller = TrackingPolicy(BC(), Modulator(), raw.shadow_hand_dof_lower_limits,
                                    raw.shadow_hand_dof_upper_limits, self.task)
        startup = sim.reset()["obs"]
        obs = startup.clone()
        controller.reset(obs)
        for _ in range(4):
            modulation = Modulator()(dict(obs=obs.clone()))["mus"].clamp(-1, 1)
            actual = controller.act(obs)
            next_obs, *_ = sim.step(modulation)
            torch.testing.assert_close(actual, raw.control, rtol=0, atol=0)
            torch.testing.assert_close(controller.tracker.observations[-1], sim.tracker.observations[-1], rtol=0, atol=0)
            controller.advance(raw.obs_buf)
            torch.testing.assert_close(controller.tracker_observation, sim.tracker_observation, rtol=0, atol=0)
            obs = next_obs["obs"]
        self.assertFalse(controller.tracker.training)
        self.assertFalse(controller.modulator.training)
        self.assertTrue(all(not p.requires_grad for p in controller.tracker.parameters()))
        torch.testing.assert_close(startup, torch.zeros_like(startup), rtol=0, atol=0)

    def test_clip_after_residual_and_enforce_two_phase_order(self):
        mean, mod = torch.full((1, 22), 1.5), torch.zeros(1, 202)
        mod[:, 180:] = -.5
        torch.testing.assert_close(combine_actions(mean, mod), torch.ones(1, 22), rtol=0, atol=0)
        model = TrackingPolicy(BC(), Modulator(), -torch.ones(22), torch.ones(22), self.task)
        obs = torch.zeros(1, 273)
        with self.assertRaises(RuntimeError):
            model.act(obs)
        model.reset(obs)
        model.act(obs)
        with self.assertRaises(RuntimeError):
            model.act(obs)
        model.advance(obs)
        with self.assertRaises(RuntimeError):
            model.advance(obs)
        model.reset(obs)
        torch.testing.assert_close(model.tracker_observation, obs)

    def test_action_step_limits_and_bc_only(self):
        lower, upper = -torch.ones(22), torch.ones(22)
        policy = TrackingPolicy(BC(), None, lower, upper, self.task)
        obs = torch.zeros(1, 273)
        policy.reset(obs)
        torch.testing.assert_close(policy.act(obs), torch.ones(1, 22))
        policy.advance(obs)
        action = torch.ones(1, 22)
        target = policy.joint_targets(torch.zeros_like(action), action)
        torch.testing.assert_close(target, torch.full_like(action, .0208375), rtol=0, atol=2e-9)
        torch.testing.assert_close(policy.joint_targets(upper[None], action), upper[None])
        self.assertAlmostEqual(1 / (self.task["sim"]["dt"] * self.task["env"]["controlFrequencyInv"]), 12, places=2)

    def test_reference_boundaries_and_observations_match_simulation(self):
        torch.testing.assert_close(reference_indices(torch.tensor([0, 298, 299]), torch.tensor([300, 300, 300])),
                                   torch.tensor([[0, 1, 2, 4, 12], [298, 299, 299, 299, 299], [299]*5]))
        spec = importlib.util.spec_from_file_location("sim2real_cpu_fixture", ROOT / "tests/test_tracking_env.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        with tempfile.TemporaryDirectory() as directory:
            module = fixture.load_cpu_environment(ROOT / "adadextrack/envs/tracking_env.py", directory)
            env = fixture.make_environment(module.TrackingEnv)
            # State features are compared with the same noise-free measurements.
            with patch("torch.randn_like", side_effect=lambda x: torch.zeros_like(x)), patch("torch.randn", side_effect=lambda shape, **kw: torch.zeros(shape, **kw)):
                env.compute_observations()
            for i in range(3):
                trajectory = int(env.env_inst_idxes2[i])
                poses = torch.stack([env.rigid_body_states[i, env.hand_body_idx_dict[name], :7]
                                     for name in ("palm", "index", "ring", "middle", "thumb")])
                reference = dict(object_pose=torch.cat((env.ref_obj_pos_batch2[trajectory], env.ref_obj_rot_batch2[trajectory]), -1),
                                 qpos=env.ref_hand_qs_batch2[trajectory])
                observation = build_observation(int(env.progress_buf[i]) - 1, env.shadow_hand_dof_pos[i], poses,
                    env.object_pose[i], reference, env.obj_latents[i],
                    env.shadow_hand_dof_lower_limits, env.shadow_hand_dof_upper_limits)
                torch.testing.assert_close(observation[0], env.obs_buf[i], rtol=0, atol=3e-7)

    def test_hand_mapping_and_native_zero(self):
        qpos = np.arange(16) / 30.
        np.testing.assert_allclose(hand_state(hand_command(qpos) + HAND_ZERO), qpos, atol=1e-15)
        self.assertAlmostEqual(hand_command(qpos)[0], qpos[1])
        self.assertAlmostEqual(hand_command(qpos)[12], qpos[4])
        with self.assertRaises(ValueError):
            hand_command(np.zeros(15))

    def test_pose_transforms_and_invalid_calibration(self):
        base = np.eye(4)
        base[:2, :2] = [[0, -1], [1, 0]]
        base[:3, 3] = [1, 2, 3]
        camera = np.eye(4)
        camera[:3, 3] = [1, 0, .1]
        np.testing.assert_allclose(object_in_base(base, camera)[:3, 3], [1, 3, 3.1])
        for invalid in (None, np.zeros((4, 4)), np.diag([2., 1., 1., 1.])):
            with self.assertRaises(ValueError):
                object_in_base(invalid, camera)

    def test_eye_to_hand_direction_and_config_export(self):
        calibration = dict(parameters=dict(eye_on_hand=False, robot_base_frame="link_base",
                                           tracking_base_frame="camera_color_optical_frame"),
                           transformation=dict(x=1., y=2., z=3., qx=0., qy=0.,
                                               qz=2**-.5, qw=2**-.5))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, config = folder / "calibration.yaml", folder / "config.yaml"
            output = folder / "deployment/config.yaml"
            source.write_text(yaml.safe_dump(calibration))
            config.write_text(yaml.safe_dump(dict(arm_ip="ROBOT_IP", base_frame="base_link",
                                                  tracker_checkpoint="checkpoints/bc_tracker_unseen_ref.pth")))
            result = load_eye_to_hand(source)
            camera_object = np.eye(4)
            camera_object[0, 3] = 1.
            np.testing.assert_allclose(object_in_base(result["T_base_camera"], camera_object)[:3, 3],
                                       [1., 3., 3.], atol=1e-15)
            export_calibration([str(source), "--config", str(config), "--output", str(output)])
            settings = yaml.safe_load(output.read_text())
            self.assertEqual(settings["base_frame"], "link_base")
            self.assertEqual(settings["camera_frame"], "camera_color_optical_frame")
            self.assertEqual(settings["arm_ip"], "ROBOT_IP")
            self.assertEqual(settings["tracker_checkpoint"], "checkpoints/bc_tracker_unseen_ref.pth")
            np.testing.assert_array_equal(settings["T_base_camera"], result["T_base_camera"])

    def test_eye_to_hand_rejects_other_modes_and_invalid_quaternion(self):
        calibration = dict(parameters=dict(eye_on_hand=False, robot_base_frame="link_base",
                                           tracking_base_frame="camera_color_optical_frame"),
                           transformation=dict(x=0., y=0., z=0., qx=0., qy=0., qz=0., qw=1.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.yaml"
            for value in (True, None, "false"):
                calibration["parameters"]["eye_on_hand"] = value
                path.write_text(yaml.safe_dump(calibration))
                with self.assertRaisesRegex(ValueError, "eye_on_hand: false"):
                    load_eye_to_hand(path)
            calibration["parameters"]["eye_on_hand"] = False
            calibration["transformation"]["qw"] = 0.
            path.write_text(yaml.safe_dump(calibration))
            with self.assertRaisesRegex(ValueError, "quaternion"):
                load_eye_to_hand(path)

    def test_xarm_pid_limits_sdk_slots_timeouts_and_stop(self):
        pd = ArmPD()
        np.testing.assert_allclose(pd.velocity(np.full(6, .1)), np.full(6, .3))
        np.testing.assert_allclose(pd.velocity(np.full(6, .2)), np.full(6, 2.))
        limited = ArmPD().velocity(np.array([2., 1., 0, 0, 0, 0]))
        np.testing.assert_allclose(limited, [2., 1., 0, 0, 0, 0])
        sdk = SDK()
        arm = XArmVelocityController(sdk)
        self.assertFalse(sdk.commands)
        arm.set_target(np.full(6, .1))
        arm._tick()
        np.testing.assert_allclose(sdk.commands[-1], [.3]*6 + [0])
        arm._target_time = time.monotonic() - 1.
        with self.assertRaises(TimeoutError):
            arm._tick()
        sdk.feedback_code = 1
        with self.assertRaises(RuntimeError):
            arm.read_position()
        sdk.feedback_code = 0
        arm.stop()
        np.testing.assert_array_equal(sdk.commands[-1], np.zeros(7))
        failing = SDK()
        failing.feedback_code = 1
        controller = XArmVelocityController(failing)
        controller.set_target(np.zeros(6))
        controller.start()
        controller._thread.join(timeout=1.)
        self.assertIsNotNone(controller.error)
        self.assertEqual(failing.mode, 4)
        np.testing.assert_array_equal(failing.commands[-1], np.zeros(7))

    def test_fk_and_limits_match_released_robot(self):
        path = ROOT / "assets/xarm_leap/urdf/xarm_leap.urdf"
        if not path.exists() or importlib.util.find_spec("pytorch_kinematics") is None:
            self.skipTest("Robot assets or optional FK dependency unavailable")
        kinematics = RobotKinematics(path)
        qpos = torch.arange(22).float() / 100
        poses = kinematics.poses(qpos)
        self.assertEqual(poses.shape, (len(POSE_LINKS), 7))
        self.assertTrue(torch.isfinite(poses).all())
        torch.testing.assert_close(kinematics.lower[6:10], torch.tensor([-.314, -1.047, -.506, -.366]))
        self.assertEqual(POLICY_JOINT_NAMES[10:14], ("joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"))
        # Named mapping equals the reference joint order used by the sim buffers.
        expected = torch.cat((qpos[:10], qpos[14:], qpos[10:14]))
        torch.testing.assert_close(qpos[kinematics.fk_order], expected, rtol=0, atol=0)

    def test_imports_do_not_load_drivers_or_simulator(self):
        subprocess.run([sys.executable, "-B", "-c", "import sys; import examples.sim2real.policy; import examples.sim2real.interfaces; "
                        "assert not any(n in sys.modules for n in ('isaacgym','rospy','xarm','leap_hand'))"],
                       cwd=str(ROOT), check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
