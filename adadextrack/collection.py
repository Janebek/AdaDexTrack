"""Collect every rollout from semantic specialists; no success filtering."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

from adadextrack.data.groups import ROOT, SOURCES, select_groups
from adadextrack.specialists import gpu_tokens, run_jobs, training_configuration, write_json

FRAMES = 300


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_configuration(job):
    task, _ = training_configuration(dict(job, epochs=1))
    task["env"].update(random_time=False, apply_lag=False, episodeLength=FRAMES, resetTime=-1)
    return task


def prepare_collection(args):
    from adadextrack.data.latents import DEFAULT_LATENTS, check_latent_metadata, latent_records
    from adadextrack.inference import build_model, check_resources
    from adadextrack.data.reference import load_reference

    groups = select_groups(SOURCES[args.dataset], args.group_id, all_groups=True, verify_hashes=True)
    entries = [entry for group in groups for entry in group]
    latents_dir = (args.latents_dir or DEFAULT_LATENTS).expanduser().resolve()
    check_resources(entries, latents_dir)
    for entry in entries:
        load_reference(entry.path)
    jobs = []
    for group in groups:
        group_id = int(group[0].group_id)
        name = "group_{:03d}".format(group_id)
        checkpoint = args.specialist_root / args.dataset / name / "best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError("Missing specialist checkpoint: " + str(checkpoint))
        model = build_model(checkpoint)
        metadata = model.adadextrack_metadata or {}
        check_latent_metadata(metadata, [entry.object_name for entry in group], latents_dir)
        expected = {entry.reference_id: entry.sha256 for entry in group}
        ids, hashes = metadata.get("reference_ids", []), metadata.get("reference_sha256", [])
        if (metadata.get("policy_type") != "specialist" or metadata.get("dataset") != args.dataset
                or str(metadata.get("group_id")) != str(group_id) or len(ids) != len(hashes)
                or len(set(map(str, ids))) != len(ids)
                or dict(zip(map(str, ids), hashes)) != expected):
            raise ValueError("Specialist checkpoint group/dataset/reference membership differs: " + str(checkpoint))
        del model
        jobs.append(dict(dataset=args.dataset, group_id=group_id, object_name=group[0].object_name,
                         manifest=str(SOURCES[args.dataset]), reference_ids=[e.reference_id for e in group],
                         reference_sha256=[e.sha256 for e in group], checkpoint=str(checkpoint),
                         checkpoint_sha256=file_digest(checkpoint), directory=str(args.output_dir / name),
                         num_envs=len(group) * args.envs_per_reference, envs_per_reference=args.envs_per_reference,
                         rounds=args.rounds, seed=args.seed + group_id, latents_dir=str(latents_dir)))
    references = [dict(reference_id=e.reference_id, group_id=e.group_id, object_name=e.object_name,
                       sha256=e.sha256) for e in entries]
    index = dict(schema_version=1, kind="bc_demonstrations", status="running", dataset=args.dataset,
                 group_ids=[job["group_id"] for job in jobs], references=references,
                 latents=latent_records([entry.object_name for entry in entries], latents_dir),
                 settings=dict(frames=FRAMES, pairs_per_episode=FRAMES-1, observation_dim=273, action_dim=22,
                               envs_per_reference=args.envs_per_reference, rounds=args.rounds, seed=args.seed,
                               worker_seed_rule="seed + group_id", initial_object_xy_noise=.05,
                               initial_object_yaw_noise_degrees=30., domain_randomization=True,
                               random_start_frame=False, action_lag=False, success_filter=False,
                               observations="Environment output, including clipping/noise; not teacher-normalized.",
                               actions="Raw teacher means; clamp to [-1,1] for execution and the BC target.",
                               frame_alignment="progress 1..299 / reference frames 0..298; terminal/reset actions excluded"),
                 teachers=[dict(group_id=j["group_id"], checkpoint=j["checkpoint"],
                                sha256=j["checkpoint_sha256"]) for j in jobs], shards=[])
    return jobs, index


def rollout_round(env, model, observation):
    """Follow reset/step exactly, retaining pre-action observations for progress 1..299."""
    import torch
    if not bool(((env.progress_buf == 0) | (env.progress_buf == FRAMES)).all()) or not bool(env.reset_buf.all()):
        raise RuntimeError("Collection round must start at a pending physical reset")
    observations, actions = [], []
    model.eval()
    with torch.no_grad():
        for frame in range(FRAMES):
            if observation.shape != (env.num_envs, 273) or not bool(torch.isfinite(observation).all()):
                raise ValueError("Invalid collection observation")
            result = model(dict(is_train=False, prev_actions=None, obs=observation, rnn_states=None))
            means = result["mus"]
            if means.shape != (env.num_envs, 22) or not bool(torch.isfinite(means).all()):
                raise ValueError("Invalid specialist action mean")
            if frame:
                if not bool((env.progress_buf == frame).all()):
                    raise RuntimeError("Observation/action frame alignment was lost")
                # Copy before env.step: some environments reuse the observation buffer.
                observations.append(observation.detach().cpu().numpy().copy())
                actions.append(means.detach().cpu().numpy().copy())
            next_obs, _, done, _ = env.step(means.clamp(-1., 1.))
            observation = next_obs["obs"]
            if not bool((env.progress_buf == frame + 1).all()):
                raise RuntimeError("Unexpected reset during collection")
            expected_done = frame == FRAMES - 1
            if bool(done.any()) != expected_done or (expected_done and not bool(done.all())):
                raise RuntimeError("Unexpected collection episode termination")
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("Invalid terminal observation")
    return np.stack(observations), np.stack(actions), observation


def save_round(job, round_index, observations, actions):
    """A shard contains one reference and all its environments for one round."""
    directory = Path(job["directory"])
    shards = []
    count = len(job["reference_ids"])
    expected = (FRAMES-1, job["num_envs"])
    if observations.shape != expected + (273,) or actions.shape != expected + (22,):
        raise ValueError("Incomplete collection round")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite collection data")
    for local_index, ref_id in enumerate(job["reference_ids"]):
        env_ids = np.arange(local_index, job["num_envs"], count, dtype=np.int64)
        repeats = len(env_ids)
        filename = "ref_{}_round_{:04d}.npz".format(ref_id, round_index)
        destination = directory / filename
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, obs=observations[:, env_ids].reshape(-1, 273).astype(np.float32),
                     actions=actions[:, env_ids].reshape(-1, 22).astype(np.float32),
                     reference_id=np.array(int(ref_id)), group_id=np.array(job["group_id"]),
                     round_index=np.array(round_index),
                     frame=np.repeat(np.arange(FRAMES-1, dtype=np.int64), repeats),
                     episode=np.tile(np.arange(repeats, dtype=np.int64) + round_index * repeats, FRAMES-1),
                     env_id=np.tile(env_ids, FRAMES-1))
        temporary.replace(destination)
        shards.append(dict(path=str(destination.relative_to(directory.parent)), reference_id=ref_id,
                           group_id=str(job["group_id"]), round_index=round_index,
                           samples=(FRAMES-1)*repeats, sha256=file_digest(destination)))
    return shards


def worker(path):
    # The coordinator may load torch on CPU; each fresh GPU worker imports Gym first.
    import isaacgym
    import torch
    import yaml
    from adadextrack.envs.tracking_env import TrackingEnv
    from adadextrack.envs.utils import set_seed
    from adadextrack.inference import build_model

    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    job = json.loads(path.read_text())
    directory = Path(job["directory"])
    env = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the collection worker")
        torch.cuda.set_device(0)
        set_seed(job["seed"])
        if file_digest(job["checkpoint"]) != job["checkpoint_sha256"]:
            raise ValueError("Specialist checkpoint changed after collection preparation")
        task = collection_configuration(job)
        os.environ["POSE_SAVE_DIR"] = task["env"]["pose_save_root"]
        (directory / "config.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
        model = build_model(job["checkpoint"], "cuda:0")
        env = TrackingEnv(task, "cuda:0", "cuda:0", 0, True, False, False)
        expected = [e.path for group in select_groups(job["manifest"], reference_ids=job["reference_ids"]) for e in group]
        if ([str(Path(p).resolve()) for p in env.traj_paths] != expected or env.num_obs != 273
                or env.num_acts != 22 or env.max_episode_length != FRAMES):
            raise RuntimeError("Collection environment does not match the requested references/dimensions")
        indices = torch.arange(job["num_envs"], device=env.device) % len(job["reference_ids"])
        if not torch.equal(env.env_inst_idxes2, indices):
            raise RuntimeError("Unexpected environment-to-reference assignment")
        obs, shards = env.reset()["obs"], []
        for round_index in range(job["rounds"]):
            observations, actions, obs = rollout_round(env, model, obs)
            shards.extend(save_round(job, round_index, observations, actions))
            del observations, actions
            write_json(directory / "status.json", dict(status="running", rounds=round_index+1))
            print("[COLLECT] Group {}: round {}/{}".format(job["group_id"], round_index+1, job["rounds"]), flush=True)
        write_json(directory / "shards.json", shards)
        write_json(directory / "status.json", dict(status="complete", rounds=job["rounds"], samples=sum(s["samples"] for s in shards)))
        return 0
    except BaseException as error:
        write_json(directory / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                   error=repr(error), group_id=job["group_id"]))
        raise
    finally:
        if env is not None:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("examples", "unseen_ref", "unseen_obj"), default="examples")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--group-id", type=int, action="append")
    selection.add_argument("--all", action="store_true", help="Collect all semantic groups in this training source")
    parser.add_argument("--latents-dir", type=Path)
    parser.add_argument("--specialist-root", type=Path, default=ROOT / "outputs/specialists",
                        help="Training output root, containing <dataset>/group_NNN/best.pth")
    parser.add_argument("--output-dir", type=Path, help="Exact collection directory (default: outputs/bc_data/<dataset>)")
    parser.add_argument("--envs-per-reference", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, nargs="+", help="Indices in CUDA_VISIBLE_DEVICES; default: first visible GPU")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.envs_per_reference < 1 or args.rounds < 1 or args.seed < 0:
        parser.error("Use positive environment/round counts and a nonnegative seed")
    args.specialist_root = args.specialist_root.expanduser().resolve()
    args.output_dir = (args.output_dir or ROOT / "outputs/bc_data" / args.dataset).expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args._worker:
        return worker(args._worker)
    gpus = gpu_tokens(args.gpus)
    if args.output_dir.exists():
        raise FileExistsError("Collection output already exists; choose a new --output-dir: " + str(args.output_dir))
    jobs, index = prepare_collection(args)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "index.json", index)
    try:
        result = run_jobs(jobs, gpus, worker_script=ROOT / "scripts/collect_specialists.py", label="COLLECT")
        if result:
            index["status"] = "failed"
        else:
            for job in jobs:
                shards = json.loads((Path(job["directory"]) / "shards.json").read_text())
                if len(shards) != len(job["reference_ids"]) * job["rounds"]:
                    raise RuntimeError("Collection did not save every reference/round")
                index["shards"].extend(shards)
            index["status"] = "complete"
            index["samples"] = sum(s["samples"] for s in index["shards"])
        write_json(args.output_dir / "index.json", index)
        print("[COLLECT] {} groups, {} references: {}. Data: {}".format(
            len(jobs), len(index["references"]), index["status"], args.output_dir), flush=True)
        return result
    except BaseException:
        index["status"] = "interrupted"
        write_json(args.output_dir / "index.json", index)
        raise
