"""CPU regression checks against outputs captured before environment cleanup."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "adadextrack/envs/tracking_env.py"


def load_cpu_environment(path, directory):
    """Load tensor logic with a stub SDK; never import Isaac Gym or create a sim."""
    utils_path = Path(directory) / "tensor_utils.py"
    utils_path.write_text(
        (ROOT / "adadextrack/envs/torch_utils.py").read_text().replace("import isaacgym\n", "")
    )
    utils_name = "tracking_test_utils_" + Path(directory).name
    spec = importlib.util.spec_from_file_location(utils_name, utils_path)
    utils = importlib.util.module_from_spec(spec)
    sys.modules[utils_name] = utils
    spec.loader.exec_module(utils)
    module = ModuleType("tracking_test_" + Path(directory).name)
    module.__file__ = str(path)
    module.__dict__.update(
        {key: getattr(utils, key) for key in dir(utils) if not key.startswith("__")}
    )
    module.VecTask = object
    module.gymtorch = SimpleNamespace(unwrap_tensor=lambda value: value)
    module.gymapi = SimpleNamespace(ENV_SPACE=0)
    sys.modules[module.__name__] = module
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (
                node.module.startswith("isaacgym")
                or node.module in ("adadextrack.envs.base", "adadextrack.envs.torch_utils")
            )
        )
    ]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def make_environment(cls):
    """Deterministic tensor states with mixed trajectory IDs and final-frame lookahead."""
    env = cls.__new__(cls)
    env.device = "cpu"
    env.sim = None
    env.gym = SimpleNamespace(
        **{
            name: lambda *args: None
            for name in (
                "refresh_dof_state_tensor",
                "refresh_actor_root_state_tensor",
                "refresh_rigid_body_state_tensor",
                "refresh_jacobian_tensors",
                "refresh_net_contact_force_tensor",
                "refresh_force_sensor_tensor",
                "refresh_dof_force_tensor",
                "set_dof_state_tensor_indexed",
                "set_dof_position_target_tensor_indexed",
                "set_actor_root_state_tensor_indexed",
            )
        }
    )
    env.num_envs = 3
    env.max_episode_length = 300
    env.num_shadow_hand_dofs = env.num_dofs = 22
    env.obs_type = "full_state"
    env.asymmetric_obs = env.apply_lag = env.repose_z = env.evaluation_save = False
    env.obs_lag_prob = 0.4
    env.progress_buf = torch.tensor([1, 299, 300])
    env.env_inst_idxes2 = torch.tensor([0, 1, 0])
    env.traj_lengths = torch.full((2,), 300, dtype=torch.long)
    env.root_state_tensor = torch.zeros(9, 13)
    env.root_state_tensor[:, 6] = 1
    env.object_indices = torch.tensor([1, 4, 7])
    env.hand_indices = torch.tensor([0, 3, 6])
    env.table_indices = torch.tensor([2, 5, 8])
    env.root_state_tensor[env.object_indices, :3] = torch.tensor([0.4, 0.0, 0.2])
    env.saved_root_tensor = env.root_state_tensor.clone()
    env.object_init_state = env.root_state_tensor[env.object_indices].clone()
    env.rigid_body_states = torch.zeros(3, 10, 13)
    env.rigid_body_states[:, :, 6] = 1
    env.rigid_body_states[:, :, :3] = torch.tensor([0.4, 0.0, 0.3])
    env.hand_body_idx_dict = dict(
        zip(
            (
                "ee_frame",
                "palm",
                "index",
                "middle",
                "ring",
                "thumb",
                "index_fsr",
                "middle_fsr",
                "ring_fsr",
                "thumb_fsr",
            ),
            range(10),
        )
    )
    env.ref_obj_pos_batch2 = torch.arange(2 * 300 * 3).reshape(2, 300, 3).float() / 1000
    env.ref_obj_rot_batch2 = torch.zeros(2, 300, 4)
    env.ref_obj_rot_batch2[:, :, 3] = 1
    env.ref_hand_qs_batch2 = torch.arange(2 * 300 * 22).reshape(2, 300, 22).float() / 10000
    env.dof_state = torch.zeros(3, 22, 2)
    env.shadow_hand_dof_pos = env.dof_state[:, :, 0]
    env.shadow_hand_dof_vel = env.dof_state[:, :, 1]
    env.shadow_hand_dof_lower_limits = torch.full((22,), -3.0)
    env.shadow_hand_dof_upper_limits = torch.full((22,), 3.0)
    env.shadow_hand_dof_default_vel = torch.zeros(22)
    env.cur_targets = torch.zeros(3, 22)
    env.prev_targets = torch.zeros(3, 22)
    env.obs_buf = torch.zeros(3, 273)
    env.prev_obs = torch.ones_like(env.obs_buf)
    env.obj_latents = torch.arange(64).float().repeat(3, 1) / 64
    env.z_theta = torch.tensor([0.1, -0.2, 0.3])
    env.hand_positions = env.root_state_tensor[:, :3]
    env.hand_orientations = env.root_state_tensor[:, 3:7]
    env.hand_linvels = env.root_state_tensor[:, 7:10]
    env.hand_angvels = env.root_state_tensor[:, 10:13]
    env.object_init_euler_xy = torch.zeros(3, 2)
    env.target_hand_rot = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(3, 1)
    env.reset_cnt = 0
    env.randomize = env.schedule_ornt_rew_coef = env.random_time = False
    env.reset_buf = torch.zeros(3, dtype=torch.bool)
    env.reset_goal_buf = torch.zeros(3, dtype=torch.bool)
    env.successes = torch.zeros(3)
    env.cfg = {"env": {"initial_object_xy_noise": 0.05, "initial_object_yaw_noise_degrees": 30.0}}
    return env


class TrackingEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.module = load_cpu_environment(TASK, cls.directory.name)
        cls.baseline = json.loads(
            (Path(__file__).parent / "fixtures/tracking_env_baseline.json").read_text()
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_reward_and_episode_flags_match_precleanup_outputs(self):
        values = self.baseline["reward_inputs"]
        inputs = {
            name: (
                torch.tensor(value["values"], dtype=getattr(torch, value["dtype"]))
                if isinstance(value, dict)
                else value
            )
            for name, value in values.items()
        }
        for case in self.baseline["rewards"]:
            with self.subTest(
                goal_cond=case["goal_cond"], orientation_weight=case["orientation_weight"]
            ):
                inputs.update(
                    goal_cond=case["goal_cond"], cur_ornt_rew_coef=case["orientation_weight"]
                )
                actual = self.module.compute_hand_reward(**inputs)
                for value, expected in zip(actual, case["outputs"]):
                    torch.testing.assert_close(
                        value, torch.tensor(expected, dtype=value.dtype), rtol=0, atol=0
                    )

    def test_observation_layout_and_random_sequence_match_precleanup_outputs(self):
        for case in self.baseline["observations"]:
            with self.subTest(lag=case["lag"], repose=case["repose"]):
                env = make_environment(self.module.TrackingEnv)
                env.apply_lag, env.repose_z = case["lag"], case["repose"]
                torch.manual_seed(42)
                env.compute_observations()
                torch.testing.assert_close(
                    env.obs_buf, torch.tensor(case["values"]), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    torch.rand(8), torch.tensor(case["next_random_values"]), rtol=0, atol=0
                )
                self.assertEqual(env.obs_buf.shape, (3, 273))
                torch.testing.assert_close(
                    env.goal_pos, env.ref_obj_pos_batch2[env.env_inst_idxes2, env.progress_buf - 1]
                )

    def test_reset_states_and_random_sequence_match_precleanup_outputs(self):
        for case in self.baseline["resets"]:
            with self.subTest(random_time=case["random_time"], xy=case["xy"], yaw=case["yaw"]):
                env = make_environment(self.module.TrackingEnv)
                env.random_time = case["random_time"]
                env.cfg["env"].update(
                    initial_object_xy_noise=case["xy"], initial_object_yaw_noise_degrees=case["yaw"]
                )
                torch.manual_seed(42)
                env.reset_idx(torch.tensor([0, 2]), torch.tensor([], dtype=torch.long))
                # Historical fixtures include the removed marker between object and table.
                old_roots = torch.tensor(case["roots"]).reshape(3, 4, 13)
                expected_roots = old_roots[:, [0, 1, 3]].reshape(9, 13)
                torch.testing.assert_close(
                    env.root_state_tensor, expected_roots, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    env.progress_buf, torch.tensor(case["progress"]), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    torch.rand(8), torch.tensor(case["next_random_values"]), rtol=0, atol=0
                )

    def test_scene_setup_keeps_legacy_numpy_random_sequence(self):
        tree = ast.parse(TASK.read_text())
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_create_envs"
        )
        sampler = next(
            node
            for node in function.body
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_"
        )
        previous = np.random.get_state()
        try:
            np.random.seed(42)
            exec(
                compile(ast.Module(body=[sampler], type_ignores=[]), str(TASK), "exec"),
                {"np": np, "self": SimpleNamespace(num_envs=3)},
            )
            np.testing.assert_array_equal(
                np.random.random(8), self.baseline["numpy_next_random_values"]
            )
        finally:
            np.random.set_state(previous)


if __name__ == "__main__":
    unittest.main()
