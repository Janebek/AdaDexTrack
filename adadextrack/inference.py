#!/usr/bin/env python3
"""Run a BC tracker on AdaDexTrack references and report the paper's tracking metrics.

Run from the configured project environment. The coordinator prepares the
references and policy, then launches one fresh Isaac Gym process per batch.

The rollout follows the existing AllAll10 reset/step convention: reset() returns
the initial zero observation, the first policy action triggers the physical reset,
and post-step progress 1..300 is scored against reference frames 0..299.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TASK = "TrackingEnv"
FRAMES = 300
OBS_DIM = 273
ACTION_DIM = 22
DEFAULT_CHECKPOINT = ROOT / "checkpoints/bc_tracker_unseen_ref.pth"
DEFAULT_CSV = ROOT / "data/adadextrack_full/manifest.csv"
DEFAULT_REFERENCES = ROOT / "data/adadextrack_full/references"

TASK_CONFIG = ROOT / "configs/tracking.yaml"
TRAIN_CONFIG = ROOT / "configs/bc_network.yaml"
LINKS = ("palm_lower", "thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head")
JOINT_INVERSE = np.argsort(list(range(10)) + list(range(14, 22)) + [10, 11, 12, 13])
ERROR_FIELDS = (
    "object_pos_error_m", "object_rot_error_rad", "wrist_pos_error_m",
    "wrist_rot_error_rad", "finger_mae_rad",
)
CSV_FIELDS = (
    "reference_index", "reference_id", "group_id", "trajectory_name", "reference_path",
    "object_name", "batch_index", "batch_seed", "num_frames",
) + ERROR_FIELDS + ("mean_error_success", "completion_rate", "completed_frames")


@dataclass(frozen=True)
class Reference:
    index: int
    reference_id: str
    group_id: str
    path: str
    object_name: str


def resolved_path(value):
    return Path(value).expanduser().resolve()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    temporary = Path(path).with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)












def check_urdf(path):
    if not path.is_file():
        raise FileNotFoundError("Missing URDF: {}".format(path))
    for mesh in ET.parse(str(path)).iter("mesh"):
        filename = mesh.get("filename", "")
        if not filename:
            raise ValueError("Empty mesh filename in {}".format(path))
        mesh_path = (path.parent / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError("Missing mesh {} (from {})".format(mesh_path, path))




from adadextrack.data.latents import DEFAULT_LATENTS, check_latent_metadata, latent_records
from adadextrack.data.dataset import read_manifest
from adadextrack.data.reference import load_reference
from adadextrack.evaluation.metrics import compute_paper_metrics, aggregate_metrics, print_summary


def select_references(args):
    entries = read_manifest(args.ref_csv, reference_ids=args.reference_id, verify_hashes=True)
    if args.group_id is not None:
        wanted = {str(group) for group in args.group_id}
        if not wanted.issubset({entry.group_id for entry in entries}):
            raise ValueError("Unknown group ID")
        entries = [entry for entry in entries if entry.group_id in wanted]
    return [Reference(index, entry.reference_id, entry.group_id, entry.path, entry.object_name)
            for index, entry in enumerate(entries)]


def check_resources(references, latents_dir=DEFAULT_LATENTS):
    import yaml
    asset = yaml.safe_load(TASK_CONFIG.read_text())["env"]["asset"]
    root = ROOT / "assets"
    check_urdf(root / asset["assetFileName"])
    if not (root / "textures/texture_stone_stone_texture_0.jpg").is_file():
        raise FileNotFoundError("Table texture is missing")
    latent_records([ref.object_name for ref in references], latents_dir)
    for name in sorted({ref.object_name for ref in references}):
        check_urdf(root / "urdf/objects/GRAB" / name / (name+".urdf"))


def canonical_model_state(state):
    """Map raw BC observation-statistic names without changing tensor values."""
    aliases = {
        "running_mean_std.running_mean_std.obs." + name: "running_mean_std." + name
        for name in ("running_mean", "running_var", "count")
    }
    normalized = {}
    for name, value in state.items():
        target = aliases.get(name, name)
        if target in normalized:
            raise ValueError("Duplicate checkpoint tensor after normalization: {}".format(target))
        normalized[target] = value
    return normalized


def make_policy(params, action_dim=ACTION_DIM):
    """Construct the shared rl_games network without loading weights or importing Gym."""
    vendor = ROOT / "third_party"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import rl_games
    from rl_games.algos_torch import model_builder
    if Path(rl_games.__file__).resolve().parent != vendor / "rl_games":
        raise RuntimeError("Expected repository rl_games, imported {}".format(rl_games.__file__))
    return model_builder.ModelBuilder().load(params).build({
        "actions_num": action_dim, "input_shape": (OBS_DIM,), "num_seqs": 1,
        "value_size": 1, "normalize_input": True, "normalize_value": True,
    })


def released_modulator_parameters(checkpoint_path):
    """Recognize original released checkpoints by SHA256, without rewriting them."""
    from adadextrack.artifacts import sha256
    registry = json.loads((ROOT / "checkpoints/checkpoints.json").read_text())
    digest = sha256(checkpoint_path)
    matches = [entry for entry in registry.values()
               if entry.get("policy_type") == "modulator" and entry.get("sha256") == digest]
    if len(matches) != 1:
        raise ValueError("Unrecognized legacy modulator checkpoint. Use a released checkpoint "
                         "or a new checkpoint with embedded AdaDexTrack metadata.")
    import yaml
    entry = matches[0]
    params = yaml.safe_load((ROOT / entry["network"]).read_text())["params"]
    metadata = dict(entry, schema_version=1, model=params["model"], network=params["network"])
    return params, metadata


def build_model(checkpoint_path, device="cpu", action_dim=ACTION_DIM):
    """Load embedded architectures or checksum-registered released policy weights."""
    import torch
    import yaml

    params = yaml.safe_load(TRAIN_CONFIG.read_text())["params"]
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    metadata = checkpoint.get("adadextrack")
    if metadata is None and action_dim == 202:
        params, metadata = released_modulator_parameters(checkpoint_path)
    if metadata is not None:
        if (metadata.get("schema_version") != 1 or metadata.get("policy_type") not in (("modulator",) if action_dim == 202 else ("specialist", "bc_tracker"))
                or metadata.get("observation_dim") != OBS_DIM or metadata.get("action_dim") != action_dim):
            raise ValueError("Unsupported policy checkpoint metadata or dimensions")
        params = {"model": metadata["model"], "network": metadata["network"]}
    state = canonical_model_state(checkpoint["model"])
    for name in ("running_mean_std.running_mean", "running_mean_std.running_var", "running_mean_std.count"):
        if name not in state:
            raise ValueError("Checkpoint is missing observation normalization: {}".format(name))
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite checkpoint tensor: {}".format(name))
    if (state["running_mean_std.running_var"] < 0).any():
        raise ValueError("Checkpoint has negative observation variance")
    model = make_policy(params, action_dim)
    model.load_state_dict(state, strict=True)
    model.adadextrack_metadata = metadata
    model.requires_grad_(False)
    return model.to(device).eval()










def worker(manifest_path, recorder_factory=None):
    # Isaac Gym must be imported before torch. Only this subprocess imports it.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import isaacgym
    from isaacgym import gymtorch
    import torch
    import yaml
    from adadextrack.envs.tracking_env import TrackingEnv
    from adadextrack.envs.utils import set_seed

    manifest = json.loads(Path(manifest_path).read_text())
    config = manifest["settings"]
    refs = [Reference(**item) for item in manifest["references"]]
    batch_dir = Path(manifest_path).parent
    for name, expected in config.get("latents", {}).items():
        actual = latent_records([name], config.get("latents_dir", DEFAULT_LATENTS))[name]
        if actual != expected:
            raise ValueError("Object latent changed after evaluation preparation: " + name)
    seed = config["seed"] + manifest["batch_index"]
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the inference worker")
    os.environ["POSE_SAVE_DIR"] = str(batch_dir / "task_logs")

    class EvaluationTask(TrackingEnv):
        def reset_idx(self, env_ids, goal_env_ids):
            super().reset_idx(env_ids, goal_env_ids)
            # Undo the parent's unconditional XY/yaw perturbations before any
            # physics step, without changing its control or observation logic.
            actors = self.object_indices[env_ids]
            self.root_state_tensor[actors] = self.object_init_state[env_ids].clone()
            self.root_state_tensor[actors, 7:13] = 0
            self.z_theta[env_ids] = 0
            indices = actors.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(indices), len(indices),
            )

    task_config = yaml.safe_load(TASK_CONFIG.read_text())
    task_config["env"].update(numEnvs=len(refs), reference_manifest=config["reference_manifest"],
                              reference_ids=[ref.reference_id for ref in refs], random_time=False,
                              apply_lag=False, repose_z=False, evaluation_save=False,
                              pose_save_root=os.environ["POSE_SAVE_DIR"], resetTime=-1, episodeLength=FRAMES)
    task_config["env"]["asset"]["assetRoot"] = str(ROOT / "assets")
    task_config["task"]["randomize"] = False
    task_config["env"]["latents_root"] = config.get("latents_dir", str(DEFAULT_LATENTS))
    if recorder_factory is not None:
        task_config["env"]["enableCameraSensors"] = True
    action_dim = 202 if config.get("policy_type") == "modulator" else ACTION_DIM
    env = None
    recorder = None
    try:
        model = build_model(config["checkpoint"], config["device"], action_dim)
        tracker = build_model(config["tracker_checkpoint"], config["device"]) if action_dim == 202 else None
        env = EvaluationTask(
            task_config, config["device"], config["device"], int(config["device"].split(":")[1]), not config["render"], False, False,
        )
        if tracker is not None:
            from adadextrack.envs.modulator import ModulatorEnv
            env = ModulatorEnv(env, tracker)
        if env.viewer is not None:
            from isaacgym import gymapi
            # Aim at the first robot using environment-local coordinates.
            env.gym.viewer_camera_look_at(
                env.viewer, env.envs[0], gymapi.Vec3(1.45, -1.2, 1.25),
                gymapi.Vec3(0.45, 0, 0.25),
            )
        actual_paths = [str(Path(path).resolve()) for path in env.traj_paths]
        if actual_paths != [ref.path for ref in refs] or env.N_traj != len(refs):
            raise RuntimeError("The environment did not load exactly the requested batch references")
        expected_indices = torch.arange(len(refs), device=env.device)
        if not torch.equal(env.env_inst_idxes2, expected_indices):
            raise RuntimeError("Expected exactly one environment per reference")
        if env.num_obs != OBS_DIM or env.num_acts != action_dim or env.max_episode_length != FRAMES:
            raise RuntimeError("Environment dimensions/episode length differ from the checkpoint protocol")
        if recorder_factory is not None:
            recorder = recorder_factory(env.gym, env.sim, env.envs)
        obs = env.reset()["obs"]
        objects, palms, joints = [], [], []
        with torch.no_grad():
            for frame in range(FRAMES):
                if not torch.isfinite(obs).all():
                    raise ValueError("Non-finite observation at frame {}".format(frame))
                prediction = model({"is_train": False, "prev_actions": None, "obs": obs, "rnn_states": None})
                action = prediction["mus"]
                if action.shape != (len(refs), action_dim) or not torch.isfinite(action).all():
                    raise ValueError("Invalid policy action at frame {}".format(frame))
                next_obs, _, done, _ = env.step(action.clamp(-1.0, 1.0))
                if recorder is not None:
                    recorder.capture(frame)
                # force_render is disabled, so step() does not draw the viewer.
                # Draw once per control step after the new physics state exists.
                if env.viewer is not None:
                    env.render()
                obs = next_obs["obs"]
                if not bool(torch.all(env.progress_buf == frame + 1)) or env.reset_cnt != 1:
                    raise RuntimeError("Rollout was reset or lost frame alignment")
                if bool(done.any()) != (frame == FRAMES - 1) or (frame == FRAMES - 1 and not bool(done.all())):
                    raise RuntimeError("Unexpected episode termination at frame {}".format(frame))
                objects.append(env.object_pose.detach().cpu().numpy().copy())
                palms.append(env.right_hand_pose.detach().cpu().numpy().copy())
                joints.append(env.shadow_hand_dof_pos.detach().cpu().numpy().copy())
                if frame == 0 or (frame + 1) % 50 == 0:
                    print("[BC-EVAL] Batch {}: {}/{} frames".format(manifest["batch_index"] + 1, frame + 1, FRAMES), flush=True)
        objects, palms, joints = np.stack(objects), np.stack(palms), np.stack(joints)
        rows = []
        for index, ref in enumerate(refs):
            metrics = compute_paper_metrics(objects[:, index], palms[:, index], joints[:, index], load_reference(ref.path))
            rows.append({
                "reference_index": ref.index, "reference_id": ref.reference_id, "group_id": ref.group_id,
                "trajectory_name": Path(ref.path).name, "reference_path": ref.path,
                "object_name": ref.object_name, "batch_index": manifest["batch_index"],
                "batch_seed": seed, **metrics,
            })
        # Keep the existing evaluator's plural metadata names and file pattern.
        np.savez_compressed(
            batch_dir / "poses_multi_ep00300.npz",
            obj_poses=objects, hand_poses=palms, robot_qpos=joints,
            env_ids=np.arange(len(refs)), traj_ids=np.arange(len(refs)),
            traj_names=np.array([Path(ref.path).name for ref in refs]),
            object_names=np.array([ref.object_name for ref in refs]),
            reference_ids=np.array([ref.reference_id for ref in refs]),
            group_ids=np.array([ref.group_id for ref in refs]),
            reference_paths=np.array([ref.path for ref in refs]),
            reference_indices=np.array([ref.index for ref in refs]),
            progress=np.arange(1, FRAMES + 1), reference_frames=np.arange(FRAMES),
            seed=np.array(seed), quaternion_order=np.array("xyzw"),
        )
        write_json(batch_dir / "metrics.json", rows)
    finally:
        try:
            if recorder is not None:
                recorder.close(require_complete=sys.exc_info()[0] is None)
        finally:
            if env is not None:
                if env.viewer is not None:
                    env.gym.destroy_viewer(env.viewer)
                env.gym.destroy_sim(env.sim)


def preflight(args):
    for path in (args.checkpoint, args.ref_csv, TASK_CONFIG, TRAIN_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.ref_base_dir.is_dir():
        raise NotADirectoryError(args.ref_base_dir)
    refs = select_references(args)
    print("[BC-EVAL] Preparing {} references and object resources...".format(len(refs)), flush=True)
    for ref in refs:
        load_reference(ref.path)
    args.reference_manifest = str(args.ref_csv)
    check_resources(refs, args.latents_dir)
    import torch

    action_dim = 202 if getattr(args, "policy_type", None) == "modulator" else ACTION_DIM
    model = build_model(args.checkpoint, action_dim=action_dim)
    latent_hashes = check_latent_metadata(model.adadextrack_metadata, [ref.object_name for ref in refs], args.latents_dir)
    paired = {}
    if action_dim == 202:
        training_dataset = model.adadextrack_metadata.get("dataset")
        if getattr(args, "dataset", training_dataset) != training_dataset:
            raise ValueError("The modulator checkpoint belongs to a different dataset partition")
        tracker = build_model(args.tracker_checkpoint)
        check_latent_metadata(tracker.adadextrack_metadata, [ref.object_name for ref in refs], args.latents_dir)
        paired["tracker_sha256"] = hashlib.sha256(args.tracker_checkpoint.read_bytes()).hexdigest()
        if model.adadextrack_metadata.get("tracker_sha256") != paired["tracker_sha256"]:
            raise ValueError("Modulator checkpoint was trained with a different BC tracker")
    if (model.adadextrack_metadata or {}).get("policy_type") == "specialist":
        group_id = str(model.adadextrack_metadata["group_id"])
        if {ref.group_id for ref in refs} != {group_id}:
            raise ValueError("A specialist checkpoint can evaluate only its own semantic group: " + group_id)
    with torch.no_grad():
        result = model({"is_train": False, "prev_actions": None, "obs": torch.zeros(2, OBS_DIM), "rnn_states": None})
    if result["mus"].shape != (2, action_dim) or not all(torch.isfinite(result[key]).all() for key in ("mus", "values")):
        raise ValueError("Checkpoint CPU forward produced invalid outputs")
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    print("[BC-EVAL] Loaded {} references, {} objects and the BC tracker.".format(
        len(refs), len({ref.object_name for ref in refs})), flush=True)
    return refs, {"checkpoint_sha256": digest.hexdigest(), "python": sys.version.split()[0], "torch": torch.__version__, "latents": latent_hashes, **paired}


def run_batches(args, refs, versions):
    started = datetime.now(timezone.utc)
    run_dir = args.output_dir / started.strftime("%Y%m%d_%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "checkpoint": str(args.checkpoint), "ref_csv": str(args.ref_csv),
        "policy_type": getattr(args, "policy_type", "tracker"),
        "tracker_checkpoint": str(getattr(args, "tracker_checkpoint", "")),
        "latents_dir": str(args.latents_dir),
        "ref_base_dir": str(args.ref_base_dir), "task": TASK,
        "reference_manifest": getattr(args, "reference_manifest", ""),
        "dataset_manifest_relative": str(args.ref_csv.relative_to(ROOT)) if ROOT in args.ref_csv.parents else None,
        "dataset_manifest_sha256": hashlib.sha256(args.ref_csv.read_bytes()).hexdigest(),
        "batch_size": args.batch_size, "device": args.device, "seed": args.seed,
        "batch_seed_rule": "seed + zero_based_batch_index", "frames": FRAMES,
        "headless": not args.render, "render": args.render, "selected_reference_ids": [ref.reference_id for ref in refs], "deterministic_policy": True, "initial_pose_perturbation": False,
        "domain_randomization": False, "lag": False,
        "observation_noise": "Existing task Gaussian noise retained (std 0.002; joints, positions, and RPY perturbations).",
        "frame_alignment": "Post-step progress 1..300 versus reference frames 0..299; existing zero-observation reset convention.",
        "thresholds": {"combined_position_m": 0.10, "combined_rotation_deg": 60, "finger_mae_deg": 40},
        **versions,
    }
    summary = {
        "status": "running", "started_at_utc": started.isoformat(), "settings": settings,
        "expected_references": len(refs), "completed_references": 0,
        "objects": len({ref.object_name for ref in refs}),
    }
    write_json(run_dir / "summary.json", summary)
    rows = []
    batches = [refs[start:start + args.batch_size] for start in range(0, len(refs), args.batch_size)]
    print("[BC-EVAL] Output: {}\n[BC-EVAL] {} batches: {}".format(run_dir, len(batches), [len(batch) for batch in batches]), flush=True)
    try:
        for batch_index, batch in enumerate(batches):
            batch_dir = run_dir / "batches" / "batch_{:03d}".format(batch_index)
            batch_dir.mkdir(parents=True)
            manifest = batch_dir / "manifest.json"
            write_json(manifest, {"settings": settings, "batch_index": batch_index, "references": [asdict(ref) for ref in batch]})
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--worker-manifest", str(manifest)]
            print("[BC-EVAL] Starting batch {}/{} ({} references)".format(batch_index + 1, len(batches), len(batch)), flush=True)
            with (batch_dir / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        if line.startswith("[BC-EVAL]"):
                            print(line, end="", flush=True)
                    code = process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                if code != 0:
                    tail = (batch_dir / "worker.log").read_text(errors="replace").splitlines()[-25:]
                    raise RuntimeError("Batch {} failed (exit {}). Log: {}\n{}".format(batch_index, code, batch_dir / "worker.log", "\n".join(tail)))
            batch_rows = json.loads((batch_dir / "metrics.json").read_text())
            aggregate_metrics(batch_rows, [ref.reference_id for ref in batch])
            rows.extend(batch_rows)
            write_rows(run_dir / "per_reference.csv", rows)
            summary["completed_references"] = len(rows)
            write_json(run_dir / "summary.json", summary)
        metrics = aggregate_metrics(rows, [ref.reference_id for ref in refs])
        summary.update({"status": "complete", "metrics": metrics, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(run_dir / "summary.json", summary)
        print_summary(metrics)
        print("[BC-EVAL] Evaluated all {} references. Results: {}".format(len(rows), run_dir), flush=True)
    except BaseException as error:
        summary.update({"status": "failed", "error": str(error), "completed_references": len(rows)})
        write_json(run_dir / "summary.json", summary)
        raise


def parse_args(argv=None, parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=resolved_path, default=DEFAULT_CHECKPOINT,
                        help="Policy weights; the entry point selects a default checkpoint")
    parser.add_argument("--latents-dir", type=resolved_path, default=DEFAULT_LATENTS)
    parser.add_argument("--ref-csv", type=resolved_path, default=DEFAULT_CSV)
    parser.add_argument("--ref-base-dir", type=resolved_path, default=DEFAULT_REFERENCES)
    parser.add_argument("--reference-id", type=int, action="append", help="Select a reference ID; repeat to select several")
    parser.add_argument("--group-id", type=int, action="append", help="Select a semantic group ID; repeat to select several")
    parser.add_argument("--render", action="store_true", help="Show the Isaac Gym viewer")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0", help="CUDA device for both simulation and policy (default: cuda:0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=resolved_path, default=ROOT / "logs/adadextrack_full/bc_eval", help="Parent directory for a new timestamped run")
    parser.add_argument("--worker-manifest", type=resolved_path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.reference_id is not None and args.group_id is not None:
        parser.error("Choose --reference-id or --group-id")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("--device must be a CUDA device such as cuda:0")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    try:
        if args.worker_manifest:
            worker(args.worker_manifest)
        else:
            refs, versions = preflight(args)
            run_batches(args, refs, versions)
    except KeyboardInterrupt:
        print("[BC-EVAL] Interrupted; completed batch outputs have been retained.", file=sys.stderr)
        return 130
    except Exception as error:
        if args.worker_manifest:
            raise
        print("[BC-EVAL] ERROR: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
