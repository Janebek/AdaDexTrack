"""Train semantic specialists sequentially, or one independent specialist per GPU."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from adadextrack.data.groups import ROOT, SOURCES, select_groups
from adadextrack.data.latents import DEFAULT_LATENTS, latent_records


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gpu_tokens(indices=None, environ=None):
    """CLI indices refer to the current visible device list, including UUID lists."""
    environ = os.environ if environ is None else environ
    indices = [0] if indices is None else indices
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("GPU indices must be unique nonnegative integers")
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(index) for index in indices]
    tokens = [value.strip() for value in visible.split(",") if value.strip()]
    if not tokens or "-1" in tokens or max(indices) >= len(tokens):
        raise ValueError("GPU index is outside CUDA_VISIBLE_DEVICES")
    return [tokens[index] for index in indices]


def training_configuration(job):
    import yaml
    config = yaml.safe_load((ROOT / "configs/specialist.yaml").read_text())
    task = yaml.safe_load((ROOT / "configs/tracking.yaml").read_text())
    task["env"].update(config["env"])
    task["task"].update(config["task"])
    task["env"].update(numEnvs=job["num_envs"], reference_manifest=job["manifest"],
                        reference_ids=job["reference_ids"], label=job["group_id"],
                        evaluation_save=False, pose_save_root=str(Path(job["directory"]) / "task_logs"))
    task["env"]["asset"]["assetRoot"] = str(ROOT / "assets")
    if job.get("latents_dir"):
        task["env"]["latents_root"] = job["latents_dir"]
    params = config["params"]
    params["seed"] = job["seed"]
    params["config"].update(num_actors=job["num_envs"], minibatch_size=job["num_envs"],
                             max_epochs=job["epochs"], device="cuda:0", label=job["group_id"],
                             full_experiment_name="ppo", train_dir=job["directory"])
    return task, params


def prepare_jobs(args):
    groups = select_groups(SOURCES[args.dataset], args.group_id,
                           all_groups=args.all or args.group_id is None, verify_hashes=True)
    # Loading arrays and assets here fails before starting any expensive GPU job.
    from adadextrack.data.reference import load_reference
    from adadextrack.inference import check_resources
    entries = [entry for group in groups for entry in group]
    for entry in entries:
        load_reference(entry.path)
    check_resources(entries, args.latents_dir or DEFAULT_LATENTS)
    jobs = []
    for group in groups:
        directory = args.output_dir / args.dataset / ("group_{:03d}".format(int(group[0].group_id)))
        job = dict(dataset=args.dataset, group_id=int(group[0].group_id), object_name=group[0].object_name,
                   manifest=str(SOURCES[args.dataset]), reference_ids=[entry.reference_id for entry in group],
                   reference_sha256=[entry.sha256 for entry in group], directory=str(directory),
                   num_envs=args.num_envs, epochs=args.epochs, seed=args.seed)
        job["latents"] = latent_records([entry.object_name for entry in group], args.latents_dir or DEFAULT_LATENTS)
        if args.latents_dir is not None:
            job["latents_dir"] = str(args.latents_dir.expanduser().resolve())
        task, params = training_configuration(job)
        # Paths, requested epoch budget and GPU assignment may change on resume.
        identity_task = copy.deepcopy(task)
        for key in ("reference_manifest", "pose_save_root"):
            identity_task["env"].pop(key, None)
        identity_task["env"]["asset"].pop("assetRoot")
        identity_params = copy.deepcopy(params)
        for key in ("train_dir", "max_epochs", "device"):
            identity_params["config"].pop(key)
        identity = dict(dataset=job["dataset"], group_id=job["group_id"],
                        reference_ids=job["reference_ids"], reference_sha256=job["reference_sha256"],
                        task=identity_task, params=identity_params)
        if args.latents_dir is not None:
            identity_task["env"].pop("latents_root")
            identity["latents"] = job["latents"]
        job["signature"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        job["resume_from"] = str(args.resume_from) if args.resume_from else None
        if directory.exists():
            previous = json.loads((directory / "job.json").read_text())
            if previous["signature"] != job["signature"]:
                raise ValueError("Existing output has incompatible settings: " + str(directory))
            status_path = directory / "status.json"
            status = json.loads(status_path.read_text()) if status_path.is_file() else {}
            if (status.get("status") == "complete" and status.get("epoch", 0) >= args.epochs
                    and all((directory / name).is_file() for name in ("last.pth", "best.pth"))):
                print("[SPECIALIST] Skip completed group {} (epoch {}).".format(job["group_id"], status["epoch"]), flush=True)
                continue
            if args.resume and job["resume_from"] is None:
                job["resume_from"] = str(directory / "last.pth")
            if job["resume_from"] is None:
                raise ValueError("Incomplete output exists; use --resume or a new --output-dir: " + str(directory))
        if job["resume_from"] is not None and not Path(job["resume_from"]).is_file():
            raise FileNotFoundError("Resume checkpoint is missing: " + job["resume_from"])
        jobs.append(job)
    return jobs


def stop_workers(active):
    for item in active:
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for item in active:
        process = item["process"]
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        item["log"].close()


def run_jobs(jobs, gpus, worker_script=None, label="SPECIALIST"):
    worker_script = worker_script or ROOT / "scripts/train_specialists.py"
    pending, active, failed = list(jobs), [], []
    try:
        while pending or active:
            for gpu in gpus:
                if not pending or any(item["gpu"] == gpu for item in active):
                    continue
                job = pending.pop(0)
                directory = Path(job["directory"])
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "job.json", job)
                write_json(directory / "status.json", dict(status="starting", group_id=job["group_id"]))
                log = (directory / "worker.log").open("a", buffering=1)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1")
                try:
                    process = subprocess.Popen([sys.executable, "-u", "-B", str(worker_script),
                                                "--_worker", str(directory / "job.json")],
                                               cwd=str(ROOT), env=environment, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    write_json(directory / "status.json", dict(status="failed", error="Failed to launch worker"))
                    raise
                active.append(dict(process=process, gpu=gpu, job=job, log=log))
                print("[{}] Group {}: {} references, GPU {}. Log: {}".format(
                    label, job["group_id"], len(job["reference_ids"]), gpu, directory / "worker.log"), flush=True)
            for item in active[:]:
                code = item["process"].poll()
                if code is None:
                    continue
                item["process"].wait()
                item["log"].close()
                active.remove(item)
                job = item["job"]
                path = Path(job["directory"]) / "status.json"
                status = json.loads(path.read_text())
                if code != 0 or status.get("status") != "complete" or status.get("epoch", 0) < job.get("epochs", 0):
                    status.update(status="failed", exit_code=code)
                    write_json(path, status)
                    failed.append(job["group_id"])
                print("[{}] Group {}: {}.".format(label, job["group_id"], status["status"]), flush=True)
            if active:
                time.sleep(0.2)
    except BaseException:
        stop_workers(active)
        for item in active:
            write_json(Path(item["job"]["directory"]) / "status.json", dict(status="interrupted"))
        raise
    if failed:
        print("[{}] Failed groups: {}. Inspect worker.log.".format(label, failed), flush=True)
    return int(bool(failed))


def worker(path):
    # This import must precede every torch import in the GPU subprocess.
    import isaacgym
    import torch
    import yaml
    from adadextrack.envs.tracking_env import TrackingEnv
    from adadextrack.envs.utils import set_seed
    sys.path.insert(0, str(ROOT / "third_party"))
    from rl_games.common import tr_helpers
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from adadextrack.specialist_agent import SpecialistAgent

    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    job = json.loads(path.read_text())
    directory = Path(job["directory"])
    task, params = training_configuration(job)
    (directory / "config.yaml").write_text(yaml.safe_dump(dict(task=task, params=params), sort_keys=False))
    metadata = dict(schema_version=1, policy_type="specialist", observation_dim=273, action_dim=22,
                    dataset=job["dataset"], group_id=job["group_id"], object_name=job["object_name"],
                    reference_ids=job["reference_ids"], reference_sha256=job["reference_sha256"],
                    model=copy.deepcopy(params["model"]), network=copy.deepcopy(params["network"]),
                    signature=job["signature"], latents=job.get("latents", {}))
    env = agent = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the specialist worker")
        torch.cuda.set_device(0)
        set_seed(job["seed"])
        os.environ["POSE_SAVE_DIR"] = task["env"]["pose_save_root"]
        env = TrackingEnv(task, "cuda:0", "cuda:0", 0, True, False, False)
        if [str(Path(p).resolve()) for p in env.traj_paths] != [entry.path for group in
                select_groups(job["manifest"], reference_ids=job["reference_ids"]) for entry in group]:
            raise RuntimeError("Environment loaded different specialist references")
        cfg = params["config"]
        cfg.update(vec_env=env, env_info={"observation_space": env.observation_space,
                                         "action_space": env.action_space, "agents": 1, "value_size": 1},
                   features={"observer": IsaacAlgoObserver()})
        cfg["reward_shaper"] = tr_helpers.DefaultRewardsShaper(**cfg["reward_shaper"])
        agent = SpecialistAgent(params, metadata, directory)
        if job["resume_from"] is not None:
            agent.restore(job["resume_from"])
        write_json(directory / "status.json", dict(status="running", group_id=job["group_id"], epoch=agent.epoch_num))
        if agent.epoch_num < job["epochs"]:
            agent.train()
        agent.save("last")
        write_json(directory / "status.json", dict(status="complete", group_id=job["group_id"], epoch=agent.epoch_num))
        return 0
    except BaseException as error:
        write_json(directory / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                   error=repr(error), group_id=job["group_id"]))
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
    parser.add_argument("--dataset", choices=("examples", "unseen_ref", "unseen_obj"), default="examples")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--group-id", type=int, action="append")
    selection.add_argument("--all", action="store_true", help="Train every semantic group in this source")
    parser.add_argument("--gpus", type=int, nargs="+", help="Indices in CUDA_VISIBLE_DEVICES (default: first visible GPU)")
    parser.add_argument("--latents-dir", type=Path, help="Use a new, consistently trained object representation")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=2001, help="Total PPO epochs, including restored epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/specialists")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete groups from their last.pth")
    parser.add_argument("--resume-from", type=Path, help="Explicit checkpoint; requires exactly one --group-id")
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.epochs < 1 or args.seed < 0:
        parser.error("Use positive environment/epoch counts and a nonnegative seed")
    if args.resume_from is not None and (args.resume or not args.group_id or len(args.group_id) != 1):
        parser.error("--resume-from requires one --group-id and cannot be combined with --resume")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.resume_from is not None:
        args.resume_from = args.resume_from.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args._worker:
        return worker(args._worker)
    gpus = gpu_tokens(args.gpus)
    jobs = prepare_jobs(args)
    print("[SPECIALIST] {} groups queued on {} GPU(s); {} PPO epochs per group.".format(
        len(jobs), len(gpus), args.epochs), flush=True)
    return run_jobs(jobs, gpus)
