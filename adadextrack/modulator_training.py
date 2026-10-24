"""Train one reference modulator over a complete training partition."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

from adadextrack.data.groups import ROOT, SOURCES
from adadextrack.data.dataset import read_manifest
from adadextrack.data.latents import DEFAULT_LATENTS, check_latent_metadata, latent_records
from adadextrack.specialists import gpu_tokens, run_jobs, training_configuration, write_json

TRACKERS = {name: ROOT / ("checkpoints/bc_tracker_" + name + ".pth")
            for name in ("unseen_ref", "unseen_obj")}


def configuration(job):
    import yaml
    task, params = training_configuration(dict(job, group_id=0))
    config = yaml.safe_load((ROOT / "configs/modulator.yaml").read_text())
    task["env"].update(config["env"])
    task["task"].update(config["task"])
    params["network"] = config["params"]["network"]
    params["config"].update(config["params"]["config"])
    return task, params


def prepare_job(args):
    from adadextrack.inference import build_model, check_resources
    from adadextrack.data.reference import load_reference
    entries = read_manifest(SOURCES[args.dataset], verify_hashes=True)
    check_resources(entries, args.latents_dir)
    for entry in entries:
        load_reference(entry.path)
    tracker = build_model(args.tracker_checkpoint)
    metadata = tracker.adadextrack_metadata or {}
    if metadata.get("policy_type") == "specialist":
        raise ValueError("Modulator initialization requires a BC tracker")
    if metadata.get("dataset", args.dataset) != args.dataset:
        raise ValueError("BC tracker was trained for a different partition")
    latents = check_latent_metadata(metadata, [e.object_name for e in entries], args.latents_dir)
    # Custom BC models must cover the entire chosen training partition.
    if metadata.get("reference_ids") is not None:
        if set(map(str, metadata["reference_ids"])) != {e.reference_id for e in entries}:
            raise ValueError("BC checkpoint training references differ from this partition")
        if metadata.get("reference_sha256") is not None:
            recorded = dict(zip(map(str, metadata["reference_ids"]), metadata["reference_sha256"]))
            if recorded != {e.reference_id: e.sha256 for e in entries}:
                raise ValueError("BC checkpoint reference contents differ from this partition")
    del tracker
    directory = args.output_dir
    job = dict(dataset=args.dataset, group_id=args.dataset, manifest=str(SOURCES[args.dataset]),
               reference_ids=[e.reference_id for e in entries], reference_sha256=[e.sha256 for e in entries],
               directory=str(directory), num_envs=args.num_envs, epochs=args.epochs, seed=args.seed,
               tracker_checkpoint=str(args.tracker_checkpoint), tracker_sha256=hashlib.sha256(args.tracker_checkpoint.read_bytes()).hexdigest(),
               latents_dir=str(args.latents_dir), latents=latents)
    task, params = configuration(job)
    identity_task, identity_params = copy.deepcopy(task), copy.deepcopy(params)
    for key in ("reference_manifest", "pose_save_root", "latents_root"):
        identity_task["env"].pop(key, None)
    identity_task["env"]["asset"].pop("assetRoot")
    for key in ("max_epochs", "train_dir", "device"):
        identity_params["config"].pop(key)
    identity = dict(dataset=args.dataset, references=dict(zip(job["reference_ids"], job["reference_sha256"])),
                    tracker_sha256=job["tracker_sha256"], latents=latents, task=identity_task, params=identity_params)
    job["signature"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    job["resume_from"] = str(args.resume_from) if args.resume_from else None
    if (directory / "job.json").exists():
        previous = json.loads((directory / "job.json").read_text())
        if previous["signature"] != job["signature"]:
            raise ValueError("Output contains a different modulator configuration")
        if args.resume:
            job["resume_from"] = str(directory / "last.pth")
        if not job["resume_from"]:
            raise ValueError("Use --resume or a new --output-dir")
    if job["resume_from"] and not Path(job["resume_from"]).is_file():
        raise FileNotFoundError(job["resume_from"])
    return job


def worker(path):
    import isaacgym  # Must precede torch.
    import torch
    import yaml
    from adadextrack.envs.tracking_env import TrackingEnv
    from adadextrack.envs.modulator import ModulatorEnv
    from adadextrack.envs.utils import set_seed
    from adadextrack.inference import build_model
    sys.path.insert(0, str(ROOT / "third_party"))
    from rl_games.common import tr_helpers
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from adadextrack.specialist_agent import SpecialistAgent

    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    job = json.loads(path.read_text())
    directory = Path(job["directory"])
    task, params = configuration(job)
    (directory / "config.yaml").write_text(yaml.safe_dump(dict(task=task, params=params), sort_keys=False))
    metadata = dict(schema_version=1, policy_type="modulator", observation_dim=273, action_dim=202,
                    dataset=job["dataset"], reference_ids=job["reference_ids"], reference_sha256=job["reference_sha256"],
                    tracker_checkpoint=job["tracker_checkpoint"], tracker_sha256=job["tracker_sha256"], latents=job["latents"],
                    model=copy.deepcopy(params["model"]), network=copy.deepcopy(params["network"]),
                    signature=job["signature"], modulation="Planner2k5: next cached tracker observation; immediate residual")
    env = agent = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(0)
        set_seed(job["seed"])
        if hashlib.sha256(Path(job["tracker_checkpoint"]).read_bytes()).hexdigest() != job["tracker_sha256"]:
            raise ValueError("BC tracker changed after preparation")
        check_latent_metadata(metadata, [e.object_name for e in read_manifest(job["manifest"])], job["latents_dir"])
        tracker = build_model(job["tracker_checkpoint"], "cuda:0")
        os.environ["POSE_SAVE_DIR"] = task["env"]["pose_save_root"]
        env = ModulatorEnv(TrackingEnv(task, "cuda:0", "cuda:0", 0, True, False, False), tracker)
        expected = [e.path for e in read_manifest(job["manifest"], reference_ids=job["reference_ids"])]
        if [str(Path(p).resolve()) for p in env.traj_paths] != expected:
            raise RuntimeError("Environment loaded different training references")
        cfg = params["config"]
        cfg.update(vec_env=env, env_info=dict(observation_space=env.observation_space, action_space=env.action_space,
                                            agents=1, value_size=1), features={"observer": IsaacAlgoObserver()})
        cfg["reward_shaper"] = tr_helpers.DefaultRewardsShaper(**cfg["reward_shaper"])
        agent = SpecialistAgent(params, metadata, directory)
        if job["resume_from"]:
            agent.restore(job["resume_from"])
        write_json(directory / "status.json", dict(status="running", epoch=agent.epoch_num))
        if agent.epoch_num < job["epochs"]:
            agent.train()
        agent.save("last")
        write_json(directory / "status.json", dict(status="complete", epoch=agent.epoch_num))
        return 0
    except BaseException as error:
        write_json(directory / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error)))
        raise
    finally:
        if agent is not None and agent.writer is not None:
            agent.writer.close()
        if env is not None:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("examples", "unseen_ref", "unseen_obj"), default="unseen_ref")
    parser.add_argument("--tracker-checkpoint", type=Path)
    parser.add_argument("--latents-dir", type=Path, default=DEFAULT_LATENTS)
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/modulator/<dataset>")
    parser.add_argument("--device", default="cuda:0", help="Single visible CUDA device")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-from", type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("Use --device cuda:N for one GPU")
    if min(args.num_envs, args.epochs) < 1 or not 0 <= args.seed < 2**32:
        parser.error("Invalid environment/epoch count or seed")
    args.tracker_checkpoint = (args.tracker_checkpoint or TRACKERS.get(args.dataset, ROOT / "outputs/bc/examples/best.pth")).expanduser().resolve()
    args.output_dir = (args.output_dir or ROOT / "outputs/modulator" / args.dataset).expanduser().resolve()
    if args.resume_from:
        args.resume_from = args.resume_from.expanduser().resolve()
    args.latents_dir = args.latents_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args._worker:
        return worker(args._worker)
    job = prepare_job(args)
    return run_jobs([job], gpu_tokens([int(args.device[5:])]), ROOT / "scripts/train_modulator.py", "MODULATOR")
