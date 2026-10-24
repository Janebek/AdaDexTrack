# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
from pathlib import Path

import numpy as np
import torch

from adadextrack.envs.torch_utils import (
    get_euler_xyz,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    scale,
    tensor_clamp,
    to_torch,
    torch_rand_float,
    unscale,
)
from adadextrack.data.dataset import read_manifest, load_task_reference_tensors
from adadextrack.control import reference_indices, relative_joint_increment
from adadextrack.envs.base import VecTask

from isaacgym import gymtorch
from isaacgym import gymapi


class TrackingEnv(VecTask):
    """Track xArm/LEAP object motions for semantic specialists and BC inference."""

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):

        self.cfg = cfg
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.label = self.cfg["env"]["label"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.synthetic_data = self.cfg["env"].get("synthetic_data", False)
        self.apply_lag = self.cfg["env"].get("apply_lag", False)
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        if self.synthetic_data:
            self.max_episode_length = 500
        else:
            self.max_episode_length = 300

        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(
                round(self.reset_time / (control_freq_inv * self.sim_params.dt))
            )
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)

        num_obs = 209 + 64
        self.num_obs_dict = {"full_state": num_obs}
        self.up_axis = "z"
        self.fingertips = ["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"]
        self.contact_sensor_names = [
            "index_tip_fsr",
            "middle_tip_fsr",
            "ring_tip_fsr",
            "thumb_tip_fsr",
        ]

        self.num_fingertips = len(self.fingertips)
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = 211
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 22

        self.dataset_references = read_manifest(
            self.cfg["env"]["reference_manifest"],
            reference_ids=self.cfg["env"].get("reference_ids"),
        )
        self.object_name = self.dataset_references[0].object_name
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if self.viewer != None:
            cam_pos = gymapi.Vec3(2.5, 1.89, 1.0)
            cam_target = gymapi.Vec3(1.1, 1.89, 0.2)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
                self.num_envs, self.num_fingertips * 6
            )

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                self.num_envs, self.num_shadow_hand_dofs + self.num_object_dofs
            )
            self.dof_force_tensor = self.dof_force_tensor[:, : self.num_shadow_hand_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_shadow_hand_dofs
        ]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1, 3)

        print(f"Contact Tensor Dimension {self.contact_tensor.shape}")

        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )
        self.total_successes = 0
        self.total_resets = 0

        def _safe_to_2d(arr, feat_dim):
            "Convert reference link arrays to float32 matrices with the expected width."
            x = np.array(arr, dtype=np.float32)
            x = np.squeeze(x)
            if x.ndim == 1:

                x = x[None, :]
            assert (
                x.ndim == 2 and x.shape[-1] == feat_dim
            ), f"Expected (*, {feat_dim}), got {x.shape}"
            return x

        def load_link_maps_for_trajs(
            traj_paths, device, T_max_main: int, traj_lengths_main: torch.Tensor
        ):
            "Pack link references by trajectory, padding short sequences with their last frame."
            N = len(traj_paths)
            assert N == int(traj_lengths_main.numel()), "Trajectory paths and lengths differ"

            all_pos_keys, all_rot_keys = set(), set()
            per_traj_pos, per_traj_rot = [], []

            for p in traj_paths:
                data = np.load(p, allow_pickle=True).item()
                pos_map = data.get("link_key_to_link_pos", {}) or {}
                rot_map = data.get("link_key_to_link_rot", {}) or {}
                per_traj_pos.append(pos_map)
                per_traj_rot.append(rot_map)
                all_pos_keys.update(list(pos_map.keys()))
                all_rot_keys.update(list(rot_map.keys()))

            pos_dict, rot_dict = {}, {}

            for k in sorted(all_pos_keys):

                pos_batch = torch.zeros(N, T_max_main, 3, device=device, dtype=torch.float32)

                for i, pmap in enumerate(per_traj_pos):
                    if k not in pmap:
                        continue

                    arr = _safe_to_2d(pmap[k], feat_dim=3)
                    T_link = int(arr.shape[0])
                    T_main = int(traj_lengths_main[i].item())
                    T_eff = min(T_link, T_main, T_max_main)

                    if T_eff > 0:
                        pos_batch[i, :T_eff, :] = torch.from_numpy(arr[:T_eff]).to(
                            device=device, dtype=torch.float32
                        )

                        if T_eff < T_max_main:
                            last = pos_batch[i, T_eff - 1 : T_eff, :].expand(T_max_main - T_eff, -1)
                            pos_batch[i, T_eff:, :] = last

                pos_dict[k] = pos_batch

            id_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)

            for k in sorted(all_rot_keys):
                rot_batch = id_quat.view(1, 1, 4).expand(N, T_max_main, 4).clone()

                for i, rmap in enumerate(per_traj_rot):
                    if k not in rmap:
                        continue

                    arr = _safe_to_2d(rmap[k], feat_dim=4)
                    T_link = int(arr.shape[0])
                    T_main = int(traj_lengths_main[i].item())
                    T_eff = min(T_link, T_main, T_max_main)

                    if T_eff > 0:
                        rot_i = torch.from_numpy(arr[:T_eff]).to(device=device, dtype=torch.float32)

                        rot_batch[i, :T_eff, :] = rot_i

                        if T_eff < T_max_main:
                            last = rot_batch[i, T_eff - 1 : T_eff, :].expand(T_max_main - T_eff, -1)
                            rot_batch[i, T_eff:, :] = last

                rot_dict[k] = rot_batch

            return pos_dict, rot_dict

        (self.tot_key_to_tot_link_pos2, self.tot_key_to_tot_link_rot2) = load_link_maps_for_trajs(
            traj_paths=self.traj_paths,
            device=self.device,
            T_max_main=self.max_episode_length + 30,
            traj_lengths_main=self.traj_lengths,
        )

        self.schedule_ornt_rew_coef = self.cfg["env"].get("schedule_ornt_rew_coef", False)
        self.lowest_ornt_rew_coef = self.cfg["env"].get("lowest_ornt_rew_coef", 0.03)
        self.highest_ornt_rew_coef = self.cfg["env"].get("highest_ornt_rew_coef", 0.33)
        self.ornt_rew_coef_warm_starting_steps = self.cfg["env"].get(
            "ornt_rew_coef_warm_starting_steps", 100
        )
        self.ornt_rew_coef_increasing_steps = self.cfg["env"].get(
            "ornt_rew_coef_increasing_steps", 200
        )
        self.cur_ornt_rew_coef = self.cfg["env"].get("cur_ornt_rew_coef", 0.00)
        self.rew_env_reset_nn = 0

        self.reset_cnt = 0

        self.actions = torch.zeros((self.num_envs, 22), dtype=torch.float, device=self.device)

        self.prev_obs = torch.zeros_like(self.obs_buf)
        self.prev_actions = torch.zeros_like(self.actions)

        lag_prob = self.cfg["env"].get("lag_prob", 0.1)
        self.obs_lag_prob = lag_prob
        self.action_lag_prob = lag_prob

        self.evaluation_save = self.cfg["env"].get("evaluation_save", False)

        self.pose_save_root = Path(
            os.path.expanduser(os.getenv("POSE_SAVE_DIR", "logs/poses_10"))
        ).resolve()
        self.pose_run_dir = self.pose_save_root / "obj_poses_multi"
        self.pose_run_dir.mkdir(parents=True, exist_ok=True)

        self.obj_pose_hist_np = []
        self.hand_pose_hist_np = []
        self.robot_qpos_hist_np = []

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):

        loaded_references = load_task_reference_tensors(self.dataset_references, self.device)

        (
            self.ref_obj_pos_batch2,
            self.ref_obj_rot_batch2,
            self.ref_hand_qs_batch2,
            self.traj_lengths,
            self.traj_mask,
            self.traj_names,
            self.traj_paths,
        ) = loaded_references

        self.N_traj = self.ref_obj_pos_batch2.shape[0]

        self.env_inst_idxes2 = (torch.arange(self.num_envs, device=self.device) % self.N_traj).to(
            torch.long
        )

        self.repose_z = self.cfg["env"]["repose_z"]
        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_time = self.cfg["env"]["random_time"]
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../../assets"
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        object_asset_file = ""
        table_texture_files = "assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get(
                "assetFileName", shadow_hand_asset_file
            )
            object_asset_file = self.cfg["env"]["asset"].get(
                "objectassetFileName", object_asset_file
            )

        def build_object_asset_path(template_relpath: str, obj_name: str) -> str:

            p = Path(template_relpath)
            new_parent = p.parent.parent / obj_name
            new_file = new_parent / f"{obj_name}{p.suffix}"
            return str(new_file)

        self.traj_object_names = [entry.object_name for entry in self.dataset_references]
        unique_obj_names = sorted(set(self.traj_object_names))

        def load_object_latent(
            urdf_path: str,
            latents_root: str = "assets/latents",
            suffix: str = "_latent.npy",
        ):
            "Load the object latent by URDF name, accepting case-insensitive filenames."
            urdf_path = Path(urdf_path)

            candidates = []
            if urdf_path.stem:
                candidates.append(urdf_path.stem)
            if urdf_path.parent.name:
                candidates.append(urdf_path.parent.name)

            seen, names = set(), []
            for n in candidates:
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)

            latents_root = Path(latents_root)
            tried_paths = []

            for name in names:
                candidate = latents_root / f"{name}{suffix}"
                tried_paths.append(str(candidate))
                if candidate.is_file():
                    return np.load(candidate, allow_pickle=False)

            all_npy = list(latents_root.glob("*.npy"))
            lower_index = {p.name.lower(): p for p in all_npy}
            for name in names:
                key = f"{name.lower()}{suffix.lower()}"
                if key in lower_index:
                    return np.load(lower_index[key], allow_pickle=False)

            raise FileNotFoundError(
                f"No object latent found. Tried: {tried_paths}; directory contains {len(all_npy)} .npy files."
            )

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 300000

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        shadow_hand_asset = self.gym.load_asset(
            self.sim, asset_root, shadow_hand_asset_file, asset_options
        )

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        coeff = 1.0

        xarm_stiffness = [185744.39, 251381.64, 231180.86, 215584.08, 242082.82, 222714.50]

        xarm_stiffness = [stiffness * coeff for stiffness in xarm_stiffness]

        xarm_damping = [71949.53, 97685.80, 89584.31, 83426.42, 93789.61, 86430.36]

        xarm_damping = [damping * coeff for damping in xarm_damping]

        for i in range(self.num_shadow_hand_dofs):
            if i < 6:

                shadow_hand_dof_props["stiffness"][i] = xarm_stiffness[i]
                shadow_hand_dof_props["damping"][i] = xarm_damping[i]
                shadow_hand_dof_props["velocity"][i] = 2.0
                shadow_hand_dof_props["armature"][i] = 0.0
                self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props["lower"][i])
                self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props["upper"][i])
            else:

                shadow_hand_dof_props["effort"][i] = 0.7
                shadow_hand_dof_props["stiffness"][i] = 20
                shadow_hand_dof_props["damping"][i] = 1
                shadow_hand_dof_props["armature"][i] = 0.01
                self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props["lower"][i])
                self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props["upper"][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        # Preserve the legacy random draws so existing seeds retain the same sequence.
        # These unused proposals never supplied the actor's actual DOF properties.
        for _ in range(self.num_envs):
            np.random.uniform(0.8, 1.2, size=6)
            np.random.uniform(0.8, 1.2, size=6)

        self.actuated_dof_indices = to_torch(
            self.actuated_dof_indices, dtype=torch.long, device=self.device
        )
        self.shadow_hand_dof_lower_limits = to_torch(
            self.shadow_hand_dof_lower_limits, device=self.device
        )
        self.shadow_hand_dof_upper_limits = to_torch(
            self.shadow_hand_dof_upper_limits, device=self.device
        )
        self.shadow_hand_dof_default_pos = to_torch(
            self.shadow_hand_dof_default_pos, device=self.device
        )
        self.shadow_hand_dof_default_vel = to_torch(
            self.shadow_hand_dof_default_vel, device=self.device
        )

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = False
        object_asset_options.disable_gravity = False
        object_asset_options.use_mesh_materials = True
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.vhacd_params.resolution = 10000
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_asset = None

        self.object_assets = {}
        self.obj_latent_dict = {}
        self.object_asset_info = {}

        max_obj_bodies = 0
        max_obj_shapes = 0

        for name in unique_obj_names:
            rel_urdf = build_object_asset_path(object_asset_file, name)
            asset = self.gym.load_asset(self.sim, asset_root, rel_urdf, object_asset_options)
            assert asset is not None, f"Failed to load object asset: {rel_urdf}"

            b = self.gym.get_asset_rigid_body_count(asset)
            s = self.gym.get_asset_rigid_shape_count(asset)
            d = self.gym.get_asset_dof_count(asset)
            max_obj_bodies = max(max_obj_bodies, b)
            max_obj_shapes = max(max_obj_shapes, s)

            self.object_assets[name] = asset
            self.object_asset_info[name] = dict(bodies=b, shapes=s, dofs=d, urdf_relpath=rel_urdf)

            lat = load_object_latent(rel_urdf, self.cfg["env"].get("latents_root", "assets/latents"))
            self.obj_latent_dict[name] = to_torch(lat, dtype=torch.float32, device=self.device)

        p = Path(object_asset_file)
        new_parent = p.parent.parent / self.object_name
        new_file = new_parent / f"{self.object_name}{p.suffix}"
        new_object_asset_file = str(new_file)

        object_asset = self.gym.load_asset(
            self.sim, asset_root, new_object_asset_file, object_asset_options
        )

        assert object_asset is not None

        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)

        table_dims = gymapi.Vec3(0.6, 0.6, 0.013)

        table_asset = self.gym.create_box(
            self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions()
        )

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0.0, 0, 0)

        max_agg_bodies = self.num_shadow_hand_bodies + max_obj_bodies + 1
        max_agg_shapes = self.num_shadow_hand_shapes + max_obj_shapes + 1

        self.shadow_hands = []
        self.envs = []
        self.object_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.object_indices = []
        self.table_indices = []

        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(shadow_hand_asset, name)
            for name in self.fingertips
        ]

        self.contact_sensor_handles = [
            self.gym.find_asset_rigid_body_index(shadow_hand_asset, name)
            for name in self.contact_sensor_names
        ]

        body_names = {
            "ee_frame": "link6",
            "palm": "palm_lower",
            "thumb": "thumb_tip_head",
            "index": "index_tip_head",
            "middle": "middle_tip_head",
            "ring": "ring_tip_head",
            "index_fsr": "index_tip_fsr",
            "middle_fsr": "middle_tip_fsr",
            "ring_fsr": "ring_tip_fsr",
            "thumb_fsr": "thumb_tip_fsr",
        }

        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(
                shadow_hand_asset, body_name
            )

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

        self.env_object_names = []
        self.obj_latent_list = []

        for i in range(self.num_envs):

            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            shadow_hand_actor = self.gym.create_actor(
                env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0
            )
            self.hand_start_states.append(
                [
                    shadow_hand_start_pose.p.x,
                    shadow_hand_start_pose.p.y,
                    shadow_hand_start_pose.p.z,
                    shadow_hand_start_pose.r.x,
                    shadow_hand_start_pose.r.y,
                    shadow_hand_start_pose.r.z,
                    shadow_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)

            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            traj_id = int(self.env_inst_idxes2[i].item())
            obj_name = self.traj_object_names[traj_id]
            object_asset = self.object_assets[obj_name]
            self.env_object_names.append(obj_name)

            p0 = self.ref_obj_pos_batch2[traj_id, 0]
            q0 = self.ref_obj_rot_batch2[traj_id, 0]

            object_start_pose = gymapi.Transform()
            object_start_pose.p = gymapi.Vec3(float(p0[0]), float(p0[1]), float(p0[2]))
            object_start_pose.r = gymapi.Quat(
                float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])
            )

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "object", i, 0, 0
            )

            self.object_init_state.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)

            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, -1, 0
            )
            self.gym.set_rigid_body_texture(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle
            )
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            table_shape_props[0].friction = 1
            object_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, shadow_hand_actor)
            finger_friction = 1
            hand_shape_props[13].friction = finger_friction
            hand_shape_props[23].friction = finger_friction
            hand_shape_props[28].friction = finger_friction
            hand_shape_props[18].friction = finger_friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, shadow_hand_actor, hand_shape_props)

            object_color = [90 / 255, 94 / 255, 173 / 255]
            self.gym.set_rigid_body_color(
                env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color)
            )
            table_color = [150 / 255, 150 / 255, 150 / 255]
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color)
            )

            hand_color = [255 / 255, 255 / 255, 255 / 255]
            self.gym.set_rigid_body_color(
                env_ptr, shadow_hand_actor, 13, gymapi.MESH_VISUAL, gymapi.Vec3(*hand_color)
            )
            self.gym.set_rigid_body_color(
                env_ptr, shadow_hand_actor, 23, gymapi.MESH_VISUAL, gymapi.Vec3(*hand_color)
            )
            self.gym.set_rigid_body_color(
                env_ptr, shadow_hand_actor, 28, gymapi.MESH_VISUAL, gymapi.Vec3(*hand_color)
            )
            self.gym.set_rigid_body_color(
                env_ptr, shadow_hand_actor, 18, gymapi.MESH_VISUAL, gymapi.Vec3(*hand_color)
            )

            self.obj_latent_list.append(self.obj_latent_dict[obj_name])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(
            self.num_envs, 13
        )
        self.fingertip_handles = to_torch(
            self.fingertip_handles, dtype=torch.long, device=self.device
        )
        self.contact_sensor_handles = to_torch(
            self.contact_sensor_handles, dtype=torch.long, device=self.device
        )
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

        try:
            self.obj_latent_per_env = torch.stack(self.obj_latent_list, dim=0)
        except Exception as e:

            raise RuntimeError(f"Object latents have mismatched shapes across objects: {e}")

        self.obj_latents = self.obj_latent_per_env.to(device=self.device, dtype=torch.float32)

    def _link_now(self, store_dict, link_key):
        "Gather each environment's current link target from its assigned trajectory."
        assert link_key in store_dict, f"link '{link_key}' not in store"
        batch = store_dict[link_key]
        D = batch.size(-1)

        traj_ids = self.env_inst_idxes2.to(torch.long).to(self.device)
        env_batch = batch[traj_ids]

        cur_idx = self.progress_buf - 1

        idx = cur_idx.view(-1, 1, 1).expand(-1, 1, D)
        out = env_batch.gather(1, idx).squeeze(1)

        return out

    def compute_reward(self, actions, id=-1):
        palm_link_pos = self._link_now(self.tot_key_to_tot_link_pos2, "palm_lower")
        thumb_link_pos = self._link_now(self.tot_key_to_tot_link_pos2, "thumb_tip_head")
        index_link_pos = self._link_now(self.tot_key_to_tot_link_pos2, "index_tip_head")
        middle_link_pos = self._link_now(self.tot_key_to_tot_link_pos2, "middle_tip_head")
        ring_link_pos = self._link_now(self.tot_key_to_tot_link_pos2, "ring_tip_head")

        palm_link_rot_q = self._link_now(self.tot_key_to_tot_link_rot2, "palm_lower")

        palm_link_rot_q = -palm_link_rot_q

        self.target_hand_pos = palm_link_pos

        self.target_hand_rot = palm_link_rot_q

        self.target_thumb_pos = thumb_link_pos

        self.target_index_pos = index_link_pos

        self.target_middle_pos = middle_link_pos

        self.target_ring_pos = ring_link_pos

        self.delta_target_hand_pos = self.right_hand_pos - self.target_hand_pos
        self.delta_target_hand_rot = quat_mul(
            self.right_hand_rot, quat_conjugate(self.target_hand_rot)
        )

        self.delta_target_thumb_pos = self.right_hand_th_pos - self.target_thumb_pos
        self.delta_target_index_pos = self.right_hand_ff_pos - self.target_index_pos
        self.delta_target_middle_pos = self.right_hand_mf_pos - self.target_middle_pos
        self.delta_target_ring_pos = self.right_hand_rf_pos - self.target_ring_pos

        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.current_successes[:],
            self.consecutive_successes[:],
        ) = compute_hand_reward(
            self.object_init_state[..., 2],
            self.delta_qpos,
            self.delta_target_hand_pos,
            self.delta_target_hand_rot,
            self.delta_target_thumb_pos,
            self.delta_target_index_pos,
            self.delta_target_middle_pos,
            self.delta_target_ring_pos,
            self.reset_buf,
            self.progress_buf,
            self.successes,
            self.current_successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_handle_pos,
            self.object_rot,
            self.goal_pos,
            self.goal_rot,
            self.right_hand_pos,
            self.right_hand_ff_pos,
            self.right_hand_mf_pos,
            self.right_hand_rf_pos,
            self.right_hand_th_pos,
            self.actions,
            self.av_factor,
            self.goal_cond,
            self.cur_ornt_rew_coef,
        )

        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["consecutive_successes"] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(
                        self.total_successes / self.total_resets
                    )
                )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos

        idx = self.hand_body_idx_dict["palm"]
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_pose = self.rigid_body_states[:, idx, 0:7]

        idx = self.hand_body_idx_dict["index"]
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]

        idx = self.hand_body_idx_dict["middle"]
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]

        idx = self.hand_body_idx_dict["ring"]
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]

        idx = self.hand_body_idx_dict["thumb"]
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]

        env_pos = self.ref_obj_pos_batch2[self.env_inst_idxes2]
        env_rot = self.ref_obj_rot_batch2[self.env_inst_idxes2]
        env_hand = self.ref_hand_qs_batch2[self.env_inst_idxes2]
        env_len = self.traj_lengths[self.env_inst_idxes2]

        Dq = env_hand.shape[-1]

        base_idx = self.progress_buf - 1
        idx_mat = reference_indices(base_idx, env_len)

        def gather_K(env_tensor, idx_matrix, last_dim):

            return env_tensor.gather(1, idx_matrix.unsqueeze(-1).expand(-1, -1, last_dim))

        pos_K = gather_K(env_pos, idx_mat, 3)
        rot_K = gather_K(env_rot, idx_mat, 4)
        hand_K = gather_K(env_hand, idx_mat, Dq)

        cur_goal_pos = pos_K[:, 0]
        cur_goal_rot = rot_K[:, 0]
        cur_hand_qpos_ref = hand_K[:, 0]

        self.goal_pos, self.goal_rot = cur_goal_pos, cur_goal_rot

        self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref

        self.nex_hand_qpos_ref1 = hand_K[:, 1]
        self.nex_hand_qpos_ref2 = hand_K[:, 2]
        self.nex_hand_qpos_ref4 = hand_K[:, 3]
        self.nex_hand_qpos_ref12 = hand_K[:, 4]

        self.goal_pos1, self.goal_rot1 = pos_K[:, 1], rot_K[:, 1]
        self.goal_pos2, self.goal_rot2 = pos_K[:, 2], rot_K[:, 2]
        self.goal_pos4, self.goal_rot4 = pos_K[:, 3], rot_K[:, 3]
        self.goal_pos12, self.goal_rot12 = pos_K[:, 4], rot_K[:, 4]

        if self.evaluation_save:

            snap = self.object_pose.detach().clone().cpu().numpy().copy()
            self.obj_pose_hist_np.append(snap)

            snap = self.right_hand_pose.detach().clone().cpu().numpy().copy()
            self.hand_pose_hist_np.append(snap)

            snap = self.shadow_hand_dof_pos.detach().clone().cpu().numpy().copy()
            self.robot_qpos_hist_np.append(snap)

            if int(self.progress_buf[0]) >= int(self.max_episode_length):
                obj_poses = np.stack(self.obj_pose_hist_np, axis=0)
                hand_poses = np.stack(self.hand_pose_hist_np, axis=0)
                robot_qpos = np.stack(self.robot_qpos_hist_np, axis=0)

                traj_ids_np = self.env_inst_idxes2.detach().cpu().numpy().astype(np.int64)

                traj_names_list = [self.traj_names[int(tid)] for tid in traj_ids_np]

                object_names_list = list(self.env_object_names)

                env_ids_np = np.arange(self.num_envs, dtype=np.int64)

                ep_step = int(self.progress_buf[0])
                out_path = self.pose_run_dir / f"poses_multi_ep{ep_step:05d}.npz"

                np.savez_compressed(
                    out_path,
                    obj_poses=obj_poses,
                    hand_poses=hand_poses,
                    robot_qpos=robot_qpos,
                    env_ids=env_ids_np,
                    traj_ids=traj_ids_np,
                    traj_names=np.array(traj_names_list, dtype=object),
                    object_names=np.array(object_names_list, dtype=object),
                )

                self.obj_pose_hist_np.clear()
                self.hand_pose_hist_np.clear()
                self.robot_qpos_hist_np.clear()
                print(f"[PoseLog] saved -> {out_path}")

        self.compute_full_state()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta),
                torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)

        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def compute_full_state(self, asymm_obs=False):

        self.get_unpose_quat()

        noise_std = 0.002

        def add_gauss(x, std=noise_std):
            return x + std * torch.randn_like(x)

        def add_rpy_noise_to_quat(q_xyzw, std=noise_std):

            N = q_xyzw.shape[0]
            rpy = torch.randn((N, 3), device=q_xyzw.device, dtype=q_xyzw.dtype) * std
            rx, ry, rz = rpy[:, 0], rpy[:, 1], rpy[:, 2]

            x_axis = torch.tensor([1.0, 0.0, 0.0], device=q_xyzw.device, dtype=q_xyzw.dtype).expand(
                N, 3
            )
            y_axis = torch.tensor([0.0, 1.0, 0.0], device=q_xyzw.device, dtype=q_xyzw.dtype).expand(
                N, 3
            )
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=q_xyzw.device, dtype=q_xyzw.dtype).expand(
                N, 3
            )

            qx = quat_from_angle_axis(rx, x_axis)
            qy = quat_from_angle_axis(ry, y_axis)
            qz = quat_from_angle_axis(rz, z_axis)

            q_noise = quat_mul(quat_mul(qz, qy), qx)

            q_noisy = quat_mul(q_noise, q_xyzw)
            q_noisy = q_noisy / torch.norm(q_noisy, dim=-1, keepdim=True).clamp_min(1e-8)
            return q_noisy

        self.obs_buf[:, 0 : self.num_shadow_hand_dofs] = unscale(
            add_gauss(self.shadow_hand_dof_pos),
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )

        self.obs_buf[:, 22:25] = add_gauss(self.right_hand_pos)
        self.obs_buf[:, 29:32] = add_gauss(self.right_hand_ff_pos)
        self.obs_buf[:, 36:39] = add_gauss(self.right_hand_rf_pos)
        self.obs_buf[:, 43:46] = add_gauss(self.right_hand_mf_pos)
        self.obs_buf[:, 50:53] = add_gauss(self.right_hand_th_pos)

        self.obs_buf[:, 25:29] = add_rpy_noise_to_quat(self.right_hand_rot)
        self.obs_buf[:, 32:36] = add_rpy_noise_to_quat(self.right_hand_ff_rot)
        self.obs_buf[:, 39:43] = add_rpy_noise_to_quat(self.right_hand_rf_rot)
        self.obs_buf[:, 46:50] = add_rpy_noise_to_quat(self.right_hand_mf_rot)
        self.obs_buf[:, 53:57] = add_rpy_noise_to_quat(self.right_hand_th_rot)

        obj_obs_start = 57
        self.obs_buf[:, obj_obs_start : obj_obs_start + 3] = self.unpose_point(
            add_gauss(self.object_pose[:, :3])
        )
        self.obs_buf[:, obj_obs_start + 3 : obj_obs_start + 7] = self.unpose_quat(
            add_rpy_noise_to_quat(self.object_pose[:, 3:7])
        )

        track_obj_goal_start = obj_obs_start + 7
        self.obs_buf[:, track_obj_goal_start : track_obj_goal_start + 3] = self.unpose_vec(
            self.goal_pos - self.object_pose[:, :3]
        )

        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

        self.obs_buf[:, track_obj_goal_start + 3 : track_obj_goal_start + 7] = quat_diff

        track_obj_goal_start = track_obj_goal_start + 7
        self.obs_buf[:, track_obj_goal_start : track_obj_goal_start + 3] = self.unpose_vec(
            self.goal_pos1
        )
        self.obs_buf[:, track_obj_goal_start + 3 : track_obj_goal_start + 7] = self.unpose_quat(
            self.goal_rot1
        )

        track_obj_goal_start = track_obj_goal_start + 7
        self.obs_buf[:, track_obj_goal_start : track_obj_goal_start + 3] = self.unpose_vec(
            self.goal_pos2
        )
        self.obs_buf[:, track_obj_goal_start + 3 : track_obj_goal_start + 7] = self.unpose_quat(
            self.goal_rot2
        )

        track_obj_goal_start = track_obj_goal_start + 7
        self.obs_buf[:, track_obj_goal_start : track_obj_goal_start + 3] = self.unpose_vec(
            self.goal_pos4
        )
        self.obs_buf[:, track_obj_goal_start + 3 : track_obj_goal_start + 7] = self.unpose_quat(
            self.goal_rot4
        )

        track_obj_goal_start = track_obj_goal_start + 7
        self.obs_buf[:, track_obj_goal_start : track_obj_goal_start + 3] = self.unpose_vec(
            self.goal_pos12
        )
        self.obs_buf[:, track_obj_goal_start + 3 : track_obj_goal_start + 7] = self.unpose_quat(
            self.goal_rot12
        )

        hand_goal_start = track_obj_goal_start + 7
        self.obs_buf[:, hand_goal_start : hand_goal_start + self.num_shadow_hand_dofs] = (
            self.delta_qpos[..., : self.num_shadow_hand_dofs]
        )

        nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
        canon_hand_qpos_ref = self.nex_hand_qpos_ref1
        unscaled_nex_hand_qpos_ref = unscale(
            canon_hand_qpos_ref,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = (
            unscaled_nex_hand_qpos_ref[..., : self.num_shadow_hand_dofs]
        )

        nex_ref_start = nex_ref_start + self.num_shadow_hand_dofs
        canon_hand_qpos_ref = self.nex_hand_qpos_ref2
        unscaled_nex_hand_qpos_ref = unscale(
            canon_hand_qpos_ref,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = (
            unscaled_nex_hand_qpos_ref[..., : self.num_shadow_hand_dofs]
        )

        nex_ref_start = nex_ref_start + self.num_shadow_hand_dofs
        canon_hand_qpos_ref = self.nex_hand_qpos_ref4
        unscaled_nex_hand_qpos_ref = unscale(
            canon_hand_qpos_ref,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = (
            unscaled_nex_hand_qpos_ref[..., : self.num_shadow_hand_dofs]
        )

        nex_ref_start = nex_ref_start + self.num_shadow_hand_dofs
        canon_hand_qpos_ref = self.nex_hand_qpos_ref12
        unscaled_nex_hand_qpos_ref = unscale(
            canon_hand_qpos_ref,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = (
            unscaled_nex_hand_qpos_ref[..., : self.num_shadow_hand_dofs]
        )

        obj_latent_start = nex_ref_start + self.num_shadow_hand_dofs
        self.obs_buf[:, obj_latent_start : obj_latent_start + 64] = self.obj_latents

        if self.apply_lag:

            valid_envs = self.progress_buf > 1
            use_lag = torch.rand(self.num_envs, device=self.device) < self.obs_lag_prob
            mask = valid_envs & use_lag
            self.obs_buf[mask] = self.prev_obs[mask]

        return

    def reset_target_pose(self, env_ids, apply_reset=False):
        # Retain the original draw shapes/order for seed compatibility.
        torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        torch_rand_float(0.3, 0.5, (len(env_ids), 1), device=self.device)
        torch_rand_float(-1.57, 1.57, (len(env_ids), 1), device=self.device)

        self.reset_goal_buf[env_ids] = 0

    def schedule_ornt_reward_coef(self, env_ids=None):
        env_ids_list = env_ids.detach().cpu().numpy().tolist()
        env_ids_list = [int(cur_idx) for cur_idx in env_ids_list]
        target_env_idx = 0
        if target_env_idx in env_ids_list:
            if self.rew_env_reset_nn <= self.ornt_rew_coef_warm_starting_steps:
                self.cur_ornt_rew_coef = self.lowest_ornt_rew_coef
            elif (
                self.rew_env_reset_nn
                <= self.ornt_rew_coef_warm_starting_steps + self.ornt_rew_coef_increasing_steps
            ):
                self.cur_ornt_rew_coef = self.lowest_ornt_rew_coef + (
                    self.highest_ornt_rew_coef - self.lowest_ornt_rew_coef
                ) * (self.rew_env_reset_nn - self.ornt_rew_coef_warm_starting_steps) / (
                    self.ornt_rew_coef_increasing_steps + self.ornt_rew_coef_warm_starting_steps
                )
            else:
                self.cur_ornt_rew_coef = self.highest_ornt_rew_coef
            self.rew_env_reset_nn += 1
            print(
                f"[INFO] rew_env_reset_nn: {self.rew_env_reset_nn}, ornt_rew_coef: {self.cur_ornt_rew_coef}"
            )

    def _hand_qpos_at(self, traj_ids: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        "Gather joint targets for the selected trajectory and frame indices."
        env_hand = self.ref_hand_qs_batch2[traj_ids]
        idxv = t_idx.view(-1, 1, 1)
        Dq = env_hand.shape[-1]
        q = env_hand.gather(1, idxv.expand(-1, 1, Dq)).squeeze(1)
        return q

    def reset_idx(self, env_ids, goal_env_ids):

        self.reset_cnt = self.reset_cnt + 1

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        if self.schedule_ornt_rew_coef:
            self.schedule_ornt_reward_coef(env_ids)

        traj_ids = self.env_inst_idxes2[env_ids].to(torch.long).to(self.device)

        t0 = torch.zeros_like(traj_ids, dtype=torch.long, device=self.device)

        qpos0 = self._hand_qpos_at(traj_ids, t0)

        self.shadow_hand_dof_pos[env_ids, :] = qpos0

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel

        self.prev_targets[env_ids, : self.num_shadow_hand_dofs] = qpos0
        self.cur_targets[env_ids, : self.num_shadow_hand_dofs] = qpos0

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices),
        )

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices),
        )

        all_indices = torch.unique(
            torch.cat(
                [
                    all_hand_indices,
                    self.object_indices[env_ids],
                    self.table_indices[env_ids],
                ]
            ).to(torch.int32)
        )

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[
            all_indices.to(torch.long), 0:3
        ]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[
            all_indices.to(torch.long), 3:7
        ]

        theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]

        new_object_rot = quat_from_euler_xyz(
            self.object_init_euler_xy[env_ids, 0], self.object_init_euler_xy[env_ids, 1], theta
        )
        prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        self.z_theta[env_ids] = prior_rot_z

        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0

        new_object_pos = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 0:3])

        xy_noise = self.cfg["env"].get("initial_object_xy_noise", 0.02)
        new_object_pos[:, 0:2] = torch.rand_like(new_object_pos[:, 0:2]) * (2 * xy_noise) - xy_noise

        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[
            env_ids
        ].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:3] = (
            self.root_state_tensor[self.object_indices[env_ids], 0:3] + new_object_pos
        )

        orig_quat = self.root_state_tensor[self.object_indices[env_ids], 3:7]

        yaw_noise = self.cfg["env"].get("initial_object_yaw_noise_degrees", 5.0)
        rand_angle = (torch.rand(orig_quat.shape[0], device=orig_quat.device) * 2 - 1) * (
            yaw_noise * 3.1415926 / 180.0
        )

        half_angle = rand_angle * 0.5
        sin_half = torch.sin(half_angle)
        cos_half = torch.cos(half_angle)
        rot_z_quat = torch.stack(
            [torch.zeros_like(sin_half), torch.zeros_like(sin_half), sin_half, cos_half], dim=-1
        )

        new_quat = quat_mul(rot_z_quat, orig_quat)

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_quat

        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13]
        )

        all_indices = torch.unique(
            torch.cat(
                [
                    all_hand_indices,
                    self.object_indices[env_ids],
                    self.table_indices[env_ids],
                ]
            ).to(torch.int32)
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(
                0, self.max_episode_length, (len(env_ids),), device=self.device
            )
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        if self.apply_lag:
            valid_envs = self.progress_buf > 0
            use_action_lag = torch.rand(self.num_envs, device=self.device) < self.action_lag_prob
            mask = valid_envs & use_action_lag
            self.actions[mask] = self.prev_actions[mask]

        if self.use_relative_control:
            delta_targets = relative_joint_increment(
                self.actions, self.shadow_hand_dof_speed_scale, self.dt, self.control_freq_inv,
            )

            cur_targets = self.prev_targets[..., :22] + delta_targets[..., :22]

            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                cur_targets,
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )

        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions[:, 6:],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
            )

            self.apply_forces[:, 1, :] = (
                self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            )
            self.apply_torque[:, 1, :] = (
                self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            )

            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.apply_forces),
                gymtorch.unwrap_tensor(self.apply_torque),
                gymapi.ENV_SPACE,
            )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[
            :, self.actuated_dof_indices
        ]

        if self.apply_lag:
            self.prev_obs[:] = self.obs_buf
            self.prev_actions[:] = self.actions

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices),
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:

            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(
            self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1]
        )
        self.gym.add_lines(
            self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1]
        )
        self.gym.add_lines(
            self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85]
        )


@torch.jit.script
def compute_hand_reward(
    object_init_z,
    delta_qpos,
    delta_target_hand_pos,
    delta_target_hand_rot,
    delta_target_thumb_pos,
    delta_target_index_pos,
    delta_target_middle_pos,
    delta_target_ring_pos,
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    object_handle_pos,
    object_rot,
    target_pos,
    target_rot,
    right_hand_pos,
    right_hand_ff_pos,
    right_hand_mf_pos,
    right_hand_rf_pos,
    right_hand_th_pos,
    actions,
    av_factor: float,
    goal_cond: bool,
    cur_ornt_rew_coef: float,
):
    """Original specialist reward and episode bookkeeping; coefficients are unchanged."""
    # Object position and quaternion-angle tracking errors.
    goal_pos_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    goal_rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    )

    goal_dist = goal_pos_dist + cur_ornt_rew_coef * goal_rot_dist

    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(
        right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist
    )

    right_hand_finger_dist = (
        torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1)
        + torch.norm(object_handle_pos - right_hand_mf_pos, p=2, dim=-1)
        + torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1)
        + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1)
    )
    right_hand_finger_dist = torch.where(
        right_hand_finger_dist >= 2.4, 2.4 + 0 * right_hand_finger_dist, right_hand_finger_dist
    )

    # Preserve the original L1 training terms and their relative weights.
    delta_qpos_value = torch.norm(delta_qpos[:, :], p=1, dim=-1)

    delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    delta_hand_rot_value = 2.0 * torch.asin(
        torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0)
    )

    delta_target_thumb_pos = torch.norm(delta_target_thumb_pos, p=1, dim=-1)
    delta_target_index_pos = torch.norm(delta_target_index_pos, p=1, dim=-1)
    delta_target_middle_pos = torch.norm(delta_target_middle_pos, p=1, dim=-1)
    delta_target_ring_pos = torch.norm(delta_target_ring_pos, p=1, dim=-1)

    delta_value = (
        1.0 * delta_hand_pos_value
        + 0.04 * delta_hand_rot_value
        + 0.1 * delta_qpos_value
        + 0.6
        * (
            delta_target_thumb_pos
            + delta_target_index_pos
            + delta_target_middle_pos
            + delta_target_ring_pos
        )
    )

    if goal_cond:
        action_penalty = torch.sum(actions**2, dim=-1)

        fingertip_heights = torch.stack(
            [
                right_hand_th_pos[:, 2],
                right_hand_ff_pos[:, 2],
                right_hand_mf_pos[:, 2],
                right_hand_rf_pos[:, 2],
            ],
            dim=1,
        )

        safe_threshold = 0.02 + 0.013
        capped_z = torch.clamp(fingertip_heights, min=1e-6, max=safe_threshold)
        log_z = torch.log(capped_z)
        height_weights = torch.full_like(fingertip_heights, 0.02)

        height_penalty = (height_weights * log_z).sum(dim=-1)

        target_flag = (
            (delta_hand_pos_value <= 0.4).int()
            + (delta_hand_rot_value <= 1.0).int()
            + (delta_qpos_value <= 6.6).int()
        )

        lowest = object_pos[:, 2]

        lift_z = object_init_z + 0.6 + 0.02 - 0.6

        flag = (
            (right_hand_finger_dist <= 0.48).int() + (right_hand_dist <= 0.12).int() + target_flag
        )

        flag2 = (right_hand_finger_dist <= 0.48).int() + (right_hand_dist <= 0.12).int()

        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 5, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)

        hand_up = torch.clamp(lowest - object_init_z, 0, 0.02)
        hand_up = 10 * hand_up

        hand_up2 = torch.zeros_like(right_hand_finger_dist)
        hand_up2 = torch.where(lowest >= lift_z, torch.where(flag2 == 2, 0.2, hand_up2), hand_up2)

        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus)

        reward = (
            -0.5 * right_hand_finger_dist
            - 1.0 * right_hand_dist
            + goal_hand_rew
            + hand_up
            + bonus
            - 0.5 * delta_value
            + hand_up2
            - 0.01 * action_penalty
            + height_penalty
        )

    else:
        # Default semantic-specialist reward.
        num_fingers = 4
        right_hand_finger_dist_th = 0.15 * num_fingers
        right_hand_dist_th = 0.15

        flag = (right_hand_finger_dist <= right_hand_finger_dist_th).int() + (
            right_hand_dist <= right_hand_dist_th
        ).int()

        inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)

        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 2, inhand_obj_pos_ornt_rew, goal_hand_rew)

        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(
            flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus
        )

        hand_dist_flag = (right_hand_dist <= right_hand_dist_th).int()
        right_hand_finger_dist = torch.where(
            hand_dist_flag == 1, right_hand_finger_dist, 0.0 * right_hand_finger_dist
        )

        reward = (
            (-0.5) * delta_value
            + (-0.3) * (right_hand_finger_dist + 2.0 * right_hand_dist)
            + goal_hand_rew
            + bonus
        )

    # Episode termination and success statistics retain the original thresholds.
    resets = reset_buf

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    current_successes = torch.where(resets, successes, current_successes)

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets
        + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes
