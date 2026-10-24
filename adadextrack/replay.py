"""Kinematic reference replay: write the recorded object pose and 22 joint positions."""

import argparse
import json
import math
import sys
from pathlib import Path
import time

import numpy as np

from adadextrack.data.dataset import read_manifest
from adadextrack.data.reference import load_reference

ROOT = Path(__file__).resolve().parents[1]
CAMERA_POSITION = (1.45, -1.2, 1.25)
CAMERA_TARGET = (.45, 0, .25)
CAMERA_FOV = 55.0


def playground_texture(size=1024):
    """Packed RGBA grid for a neutral floor, generated without image assets."""
    y, x = np.indices((size, size))
    checker = (x // 16 + y // 16) % 2
    pixels = np.empty((size, size, 4), dtype=np.uint8)
    pixels[..., :3] = np.where(checker[..., None], (210, 218, 228), (195, 204, 216))
    pixels[(x % 16 == 0) | (y % 16 == 0), :3] = (182, 192, 205)
    pixels[..., 3] = 255
    return pixels.reshape(size, size * 4)


def select_entries(args):
    from adadextrack.data.groups import select_groups
    groups = select_groups(args.manifest,
        group_ids=[args.group_id] if args.group_id is not None else None,
        all_groups=args.all, reference_ids=args.reference_id, verify_hashes=True)
    entries = [entry for group in groups for entry in group]
    if getattr(args, "mode", "group") == "batch":
        selected = {entry.reference_id for entry in entries}
        # Reference batches retain manifest order (or an explicit debug selection).
        entries = [entry for entry in read_manifest(args.manifest, reference_ids=args.reference_id)
                   if entry.reference_id in selected]
    return entries


def replay(args, entries=None, recorder_factory=None):
    import isaacgym
    from isaacgym import gymapi, gymtorch
    import torch

    if entries is None:
        entries = select_entries(args)
    if getattr(args, "mode", "group") == "group" and len({entry.group_id for entry in entries}) != 1:
        raise ValueError("Replay workers must contain exactly one semantic group")
    targets = [load_reference(entry.path) for entry in entries]
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1.0 / 60.0
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = True
    params.physx.num_position_iterations = 8
    device_id = int(args.device.split(":")[1])
    graphics_id = device_id if not args.headless or args.snapshot is not None or recorder_factory is not None else -1
    sim = gym.create_sim(device_id, graphics_id, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")
    viewer = None
    recorder = None
    try:
        asset_root = ROOT / "assets"
        floor = floor_texture = None
        if graphics_id >= 0:
            floor_options = gymapi.AssetOptions()
            floor_options.fix_base_link = True
            floor_options.disable_gravity = True
            floor_size = max(24.0, 4.0 * math.sqrt(len(entries)))
            floor = gym.create_box(sim, floor_size, floor_size, .04, floor_options)
            floor_texture = gym.create_texture_from_buffer(sim, 1024, 1024, playground_texture())
            gym.set_light_parameters(sim, 0, gymapi.Vec3(.65, .65, .62),
                                     gymapi.Vec3(.35, .37, .40), gymapi.Vec3(.7, -.6, 1.5))
            gym.set_light_parameters(sim, 1, gymapi.Vec3(.20, .22, .25),
                                     gymapi.Vec3(0, 0, 0), gymapi.Vec3(-1, 1, 1))
            for light in (2, 3):
                gym.set_light_parameters(sim, light, gymapi.Vec3(0, 0, 0),
                                         gymapi.Vec3(0, 0, 0), gymapi.Vec3(0, 0, 1))
        robot_options = gymapi.AssetOptions()
        robot_options.fix_base_link = True
        robot_options.disable_gravity = True
        robot_options.collapse_fixed_joints = False
        robot_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        robot = gym.load_asset(sim, str(asset_root), "xarm_leap/urdf/xarm_leap.urdf", robot_options)
        if robot is None or gym.get_asset_dof_count(robot) != 22:
            raise ValueError("Expected the 22-DoF xArm and LEAP hand asset")
        props = gym.get_asset_dof_properties(robot)
        props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        props["stiffness"].fill(0)
        props["damping"].fill(0)
        props["friction"].fill(0)
        obj_options = gymapi.AssetOptions()
        obj_options.disable_gravity = True
        obj_options.collapse_fixed_joints = True
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table = gym.create_box(sim, .6, .6, .013, table_options)
        objects, envs = {}, []
        indices = []
        lower, upper = gymapi.Vec3(-.75, -.75, 0), gymapi.Vec3(.75, .75, .75)
        for index, entry in enumerate(entries):
            env = gym.create_env(sim, lower, upper, max(1, int(math.sqrt(len(entries)))))
            envs.append(env)
            if floor is not None and (index == 0 or recorder_factory is not None):
                # Viewer replay shares one floor. Per-environment recording
                # cameras each need their own non-colliding visual floor.
                floor_pose = gymapi.Transform()
                floor_pose.p = gymapi.Vec3(0, 0, -.02)
                floor_actor = gym.create_actor(env, floor, floor_pose, "playground_floor", index, 1)
                gym.set_rigid_body_color(env, floor_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))
                gym.set_rigid_body_texture(env, floor_actor, 0, gymapi.MESH_VISUAL, floor_texture)
            # All replay actors share a collision filter so contacts cannot alter
            # the explicitly supplied reference motion.
            hand = gym.create_actor(env, robot, gymapi.Transform(), "robot", index, 1)
            gym.set_actor_dof_properties(env, hand, props)
            if entry.object_name not in objects:
                objects[entry.object_name] = gym.load_asset(sim, str(asset_root),
                    "urdf/objects/GRAB/{0}/{0}.urdf".format(entry.object_name), obj_options)
            asset = objects[entry.object_name]
            if asset is None or gym.get_asset_dof_count(asset) != 0:
                raise ValueError("Expected a rigid object asset: " + entry.object_name)
            obj = gym.create_actor(env, asset, gymapi.Transform(), "object", index, 1)
            indices.append(gym.get_actor_index(env, obj, gymapi.DOMAIN_SIM))
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(.5, 0, .0065)
            table_actor = gym.create_actor(env, table, pose, "table", index, 1)
            gym.set_rigid_body_color(env, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.65,.65,.65))
        gym.prepare_sim(sim)
        roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        dofs = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(len(entries),22,2)
        object_indices = torch.tensor(indices, dtype=torch.long)
        index32 = object_indices.to(torch.int32)
        camera_position, camera_target = CAMERA_POSITION, CAMERA_TARGET
        if args.overview:
            origins = np.array([(gym.get_env_origin(env).x, gym.get_env_origin(env).y,
                                 gym.get_env_origin(env).z) for env in envs])
            center = (origins.min(axis=0) + origins.max(axis=0)) / 2
            center += np.array(CAMERA_TARGET)
            extent = max(float(np.ptp(origins[:, 0])), float(np.ptp(origins[:, 1])))
            distance = max(2.0, 1.2 * (extent + 1.5))
            camera_target = tuple(center - origins[0])
            camera_position = tuple(center - origins[0] + distance * np.array((.75, -1.1, 1.0)))
        camera = None
        if not args.headless:
            viewer_props = gymapi.CameraProperties()
            viewer_props.horizontal_fov = CAMERA_FOV
            viewer = gym.create_viewer(sim, viewer_props)
            if viewer is None:
                raise RuntimeError("Could not create viewer")
            gym.viewer_camera_look_at(viewer, envs[0], gymapi.Vec3(*camera_position), gymapi.Vec3(*camera_target))
        if args.snapshot is not None:
            camera_props = gymapi.CameraProperties()
            camera_props.width, camera_props.height = 960, 720
            camera_props.horizontal_fov = CAMERA_FOV
            camera = gym.create_camera_sensor(envs[0], camera_props)
            gym.set_camera_location(camera, envs[0], gymapi.Vec3(*CAMERA_POSITION), gymapi.Vec3(*CAMERA_TARGET))
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if recorder_factory is not None:
            recorder = recorder_factory(gym, sim, envs)
        obj_history, joint_history = [], []
        max_obj_error = max_joint_error = 0.0
        total_frames = 300 * args.loops
        print("[REPLAY] Playing {} references, {} loops, {} fps.".format(len(entries), args.loops, args.fps), flush=True)
        for step in range(total_frames):
            if viewer is not None and gym.query_viewer_has_closed(viewer):
                break
            frame = step % 300
            obj_target = torch.tensor(np.stack([target["object_pose"][frame] for target in targets]), dtype=torch.float32)
            joint_target = torch.tensor(np.stack([target["qpos"][frame] for target in targets]), dtype=torch.float32)
            # Advance PhysX first, then write the raw reference for display.
            # Otherwise joint-limit constraints can project recorded
            # reference angles before they are rendered or logged.
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            roots[object_indices,:7] = obj_target
            roots[object_indices,7:] = 0
            dofs[:,:,0] = joint_target
            dofs[:,:,1] = 0
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(roots),gymtorch.unwrap_tensor(index32),len(indices))
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dofs))
            gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            actual_obj = roots[object_indices,:7].clone()
            actual_joints = dofs[:,:,0].clone()
            max_obj_error = max(max_obj_error, float((actual_obj-obj_target).abs().max()))
            max_joint_error = max(max_joint_error, float((actual_joints-joint_target).abs().max()))
            obj_history.append(actual_obj.numpy().copy())
            joint_history.append(actual_joints.numpy().copy())
            if recorder is not None:
                recorder.capture(step)
            if viewer is not None or camera is not None:
                gym.step_graphics(sim)
                if viewer is not None:
                    gym.draw_viewer(viewer, sim, False)
                if camera is not None and step == args.snapshot_frame:
                    gym.render_all_camera_sensors(sim)
                    gym.write_camera_image_to_file(sim,envs[0],camera,gymapi.IMAGE_COLOR,str(args.snapshot))
            if (step + 1) % 300 == 0:
                print("[REPLAY] Completed loop {}/{}.".format((step + 1) // 300, args.loops), flush=True)
            if viewer is not None and args.fps > 0:
                time.sleep(1.0 / args.fps)
        if not obj_history and args.headless:
            raise RuntimeError("Replay ended before recording any frame")
        if args.headless and len(obj_history) != total_frames:
            raise RuntimeError("Incomplete headless replay")
        if max_obj_error > 1e-4 or max_joint_error > 1e-4:
            raise RuntimeError("Reference states changed during kinematic replay: object={}, joints={}".format(max_obj_error,max_joint_error))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(str(args.output),
                            obj_poses=np.stack(obj_history) if obj_history else np.empty((0,len(entries),7),dtype=np.float32),
                            robot_qpos=np.stack(joint_history) if joint_history else np.empty((0,len(entries),22),dtype=np.float32),
                            reference_ids=np.array([entry.reference_id for entry in entries]),
                            group_ids=np.array([entry.group_id for entry in entries]),
                            reference_frames=np.arange(len(obj_history))%300, mode=np.array("kinematic_reference_replay"))
        report = dict(status="complete" if len(obj_history) == total_frames else "cancelled",
                      references=len(entries),frames=len(obj_history),max_object_state_error=max_obj_error,
                      max_joint_error=max_joint_error,output=str(args.output),snapshot=str(args.snapshot) if args.snapshot else None)
        args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
        print("[REPLAY] " + json.dumps(report),flush=True)
    finally:
        try:
            if recorder is not None:
                recorder.close(require_complete=sys.exc_info()[0] is None)
        finally:
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)


def main(argv=None):
    from adadextrack.data.groups import SOURCES, source_manifest, partition_groups
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", choices=tuple(SOURCES), help="Data source (default: examples)")
    source.add_argument("--manifest", type=Path, help="Custom reference manifest")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--reference-id", type=int, action="append", help="Select individual references for debugging")
    selection.add_argument("--group-id", type=int)
    selection.add_argument("--all", action="store_true", help="Replay all references in the selected data source")
    parser.add_argument("--mode", choices=("group", "batch"), default="group",
                        help="One semantic group per window (default), or fixed-size reference batches")
    parser.add_argument("--batch-size", type=int, help="References per window in --mode batch (default: 256)")
    parser.add_argument("--overview", action="store_true", help="Frame all references in the current window")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--loops", type=int, default=1, help="Playback loops per group or batch")
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-frame", type=int, default=150)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path, help="NPZ filename for a single selected group or batch")
    outputs.add_argument("--output-dir", type=Path, help="New directory for recordings and logs")
    parser.add_argument("--_group-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._group_config is not None:
        from adadextrack.replay_batch import load_worker_args
        replay(load_worker_args(args._group_config))
        return 0
    if args.batch_size is not None and (args.batch_size < 1 or args.mode != "batch"):
        parser.error("--batch-size requires --mode batch and a positive size")
    if args.batch_size is None:
        args.batch_size = 256
    if (args.loops < 1 or not math.isfinite(args.fps) or args.fps < 0
            or not 0 <= args.snapshot_frame < 300 or not args.device.startswith("cuda:")
            or not args.device[5:].isdigit()):
        parser.error("Use positive loops, nonnegative FPS, snapshot frame 0..299 and a cuda:N device")
    args.manifest = source_manifest(args.dataset, args.manifest)
    entries = select_entries(args)
    from adadextrack.replay_batch import partition_batches, run_groups, run_batches
    partitions = partition_groups(entries) if args.mode == "group" else partition_batches(entries, args.batch_size)
    if args.output is not None and len(partitions) != 1:
        parser.error("--output requires exactly one group or batch; otherwise use --output-dir")
    return run_groups(args, entries) if args.mode == "group" else run_batches(args, entries)
