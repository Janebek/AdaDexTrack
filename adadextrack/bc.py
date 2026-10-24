"""Train one offline BC tracker from all selected semantic specialists."""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from adadextrack.collection import file_digest
from adadextrack.data.groups import ROOT, SOURCES, select_groups
from adadextrack.specialists import write_json


class ShardStore:
    """Bounded CPU cache; even a full corpus never occupies GPU memory at once."""
    def __init__(self, root, settings, cache_bytes):
        self.root, self.settings = Path(root), settings
        self.limit, self.used, self.cache = cache_bytes, 0, OrderedDict()

    def path(self, shard):
        relative = Path(shard["path"])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or self.root.resolve() not in path.parents:
            raise ValueError("Shard path escapes collection directory")
        return path

    def get(self, shard):
        import torch
        key = shard["path"]
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        path = self.path(shard)
        with np.load(str(path), allow_pickle=False) as data:
            obs, actions = data["obs"], data["actions"]
            repeats = self.settings["envs_per_reference"]
            rows = 299 * repeats
            if (obs.shape != (rows, 273) or actions.shape != (rows, 22)
                    or obs.dtype != np.float32 or actions.dtype != np.float32
                    or not np.isfinite(obs).all() or not np.isfinite(actions).all()
                    or int(data["reference_id"]) != int(shard["reference_id"])
                    or int(data["group_id"]) != int(shard["group_id"])
                    or int(data["round_index"]) != shard["round_index"]
                    or not np.array_equal(data["frame"], np.repeat(np.arange(299), repeats))
                    or not np.array_equal(data["episode"], np.tile(
                        np.arange(repeats) + shard["round_index"] * repeats, 299))
                    or data["env_id"].shape != (rows,)
                    or not np.array_equal(data["env_id"].reshape(299, repeats),
                                          np.tile(data["env_id"][:repeats], (299, 1)))
                    or len(set(data["env_id"][:repeats].tolist())) != repeats):
                raise ValueError("Invalid observations/actions or rollout metadata: " + str(path))
        value = torch.from_numpy(obs), torch.from_numpy(actions)
        size = obs.nbytes + actions.nbytes
        while self.cache and self.used + size > self.limit:
            _, removed = self.cache.popitem(last=False)
            self.used -= sum(t.numel()*t.element_size() for t in removed)
        if size <= self.limit:
            self.cache[key] = value
            self.used += size
        return value


def read_collection(directory, dataset, cache_mb=256):
    directory = Path(directory)
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    if (index.get("schema_version") != 1 or index.get("kind") != "bc_demonstrations"
            or index.get("status") != "complete" or index.get("dataset") != dataset):
        raise ValueError("BC requires a complete collection from the selected dataset")
    settings = index["settings"]
    if (settings.get("observation_dim") != 273 or settings.get("action_dim") != 22
            or settings.get("frames") != 300 or settings.get("pairs_per_episode") != 299
            or settings.get("success_filter") is not False
            or not isinstance(settings.get("rounds"), int) or settings["rounds"] < 1
            or not isinstance(settings.get("envs_per_reference"), int) or settings["envs_per_reference"] < 1):
        raise ValueError("Unsupported demonstration settings/dimensions")
    group_ids = index["group_ids"]
    if not group_ids:
        raise ValueError("Empty specialist collection")
    groups = select_groups(SOURCES[dataset], group_ids, all_groups=True, verify_hashes=True)
    expected_refs = {e.reference_id: (e.group_id, e.sha256, e.object_name) for group in groups for e in group}
    refs = index["references"]
    actual_refs = {r["reference_id"]: (r["group_id"], r["sha256"], r["object_name"]) for r in refs}
    if actual_refs != expected_refs or len(refs) != len(expected_refs):
        raise ValueError("Demonstrations do not match the selected training-group membership")
    expected = {(ref_id, round_index) for ref_id in expected_refs for round_index in range(settings["rounds"])}
    seen, paths, sample_count = set(), set(), 0
    store = ShardStore(directory, settings, int(cache_mb * 1024 * 1024))
    for shard in index["shards"]:
        key = shard["reference_id"], shard["round_index"]
        if (key not in expected or key in seen or shard["path"] in paths
                or shard["group_id"] != expected_refs[key[0]][0]
                or shard["samples"] != 299 * settings["envs_per_reference"]):
            raise ValueError("Missing/duplicate or mismatched reference/round shard")
        if file_digest(store.path(shard)) != shard["sha256"]:
            raise ValueError("Demonstration shard checksum mismatch: " + shard["path"])
        store.get(shard)  # Check every shard, including finite values, before any optimizer step.
        seen.add(key)
        paths.add(shard["path"])
        sample_count += shard["samples"]
    if seen != expected or index.get("samples") != sample_count:
        raise ValueError("Collection is incomplete or has an incorrect sample count")
    return index, store, file_digest(index_path)


def iter_batches(shards, store, batch_size, seed):
    """Visit every row once, shuffle shards/rows, and merge a singleton final batch."""
    import torch
    order = list(shards)
    random.Random(seed).shuffle(order)
    generator = torch.Generator().manual_seed(seed)
    pending_obs, pending_actions, count, previous = [], [], 0, None
    for shard in order:
        obs, actions = store.get(shard)
        permutation = torch.randperm(len(obs), generator=generator)
        cursor = 0
        while cursor < len(obs):
            take = min(batch_size - count, len(obs) - cursor)
            indices = permutation[cursor:cursor+take]
            pending_obs.append(obs[indices])
            pending_actions.append(actions[indices])
            cursor += take
            count += take
            if count == batch_size:
                current = torch.cat(pending_obs), torch.cat(pending_actions)
                if previous is not None:
                    yield previous
                previous = current
                pending_obs, pending_actions, count = [], [], 0
    if count:
        tail = torch.cat(pending_obs), torch.cat(pending_actions)
        if count == 1 and previous is not None:
            previous = tuple(torch.cat([a, b]) for a, b in zip(previous, tail))
        else:
            if previous is not None:
                yield previous
            previous = tail
    if previous is not None:
        yield previous


def action_loss(predicted, teacher_means):
    return (predicted - teacher_means.clamp(-1., 1.)).square().sum(dim=-1).mean()


def save_checkpoint(path, state):
    import torch
    path = Path(path)
    temporary = path.with_suffix(".pth.tmp")
    torch.save(state, str(temporary))
    temporary.replace(path)


def train(args):
    import torch
    import yaml
    from adadextrack.inference import make_policy

    if args.resume:
        for name in ("last.pth", "best.pth"):
            if not (args.output_dir / name).is_file():
                raise FileNotFoundError("BC resume requires " + str(args.output_dir / name))
    elif args.output_dir.exists():
        raise FileExistsError("BC output already exists; use --resume or a new --output-dir")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a valid --device")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    index, store, data_digest = read_collection(args.data_dir, args.dataset, args.cache_mb)
    if args.latents_dir is not None:
        from adadextrack.data.latents import check_latent_metadata
        if not index.get("latents"):
            raise ValueError("Collection has no latent provenance; collect it again before using --latents-dir")
        check_latent_metadata(index, [ref["object_name"] for ref in index["references"]], args.latents_dir)
    params = yaml.safe_load(args.network_config.read_text())["params"]
    identity = dict(dataset=args.dataset, data_sha256=data_digest, model=params["model"], network=params["network"],
                    batch_size=args.batch_size, learning_rate=args.learning_rate,
                    max_grad_norm=args.max_grad_norm, seed=args.seed, weight_decay=0.)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    metadata = dict(schema_version=1, policy_type="bc_tracker", observation_dim=273, action_dim=22,
                    dataset=args.dataset, group_ids=index["group_ids"], latents=index.get("latents", {}),
                    reference_ids=[r["reference_id"] for r in index["references"]],
                    reference_sha256=[r["sha256"] for r in index["references"]],
                    model=params["model"], network=params["network"], signature=signature,
                    data_sha256=data_digest, selection_metric="mean_training_action_MSE_sum_over_22_dimensions")
    model = make_policy(params).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(str(args.output_dir / "last.pth"), map_location="cpu", weights_only=False)
        if checkpoint.get("adadextrack", {}).get("signature") != signature:
            raise ValueError("BC resume data/network/training settings differ from the saved checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_loss = checkpoint["epoch"], checkpoint["best_loss"]
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if args.device.startswith("cuda") and rng["cuda"]:
            torch.cuda.set_rng_state(rng["cuda"][0].cpu(), device=args.device)
    else:
        args.output_dir.mkdir(parents=True)
    config = dict(identity, epochs=args.epochs, device=args.device, data_dir=str(args.data_dir),
                  samples=index["samples"], groups=len(index["group_ids"]), references=len(index["references"]),
                  cache_mb=args.cache_mb, max_cached_data_bytes=store.limit)
    write_json(args.output_dir / "config.json", config)
    print("[BC-TRAIN] {} groups, {} references, {} samples; {} -> {} epochs on {}".format(
        config["groups"], config["references"], config["samples"], start_epoch, args.epochs, args.device), flush=True)
    try:
        with (args.output_dir / "metrics.jsonl").open("a") as log:
            for epoch in range(start_epoch + 1, args.epochs + 1):
                model.train()
                loss_sum, rows, steps = 0., 0, 0
                for observations, actions in iter_batches(index["shards"], store, args.batch_size, args.seed+epoch):
                    observations, actions = observations.to(args.device), actions.to(args.device)
                    result = model(dict(is_train=True, prev_actions=actions.clamp(-1., 1.),
                                        obs=observations, rnn_states=None))
                    loss = action_loss(result["mus"], actions)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Non-finite BC loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(observations)
                    rows += len(observations)
                    steps += 1
                if rows != index["samples"]:
                    raise RuntimeError("An epoch did not visit every demonstration exactly once")
                mean_loss = loss_sum / rows
                improved = mean_loss < best_loss
                best_loss = min(best_loss, mean_loss)
                # Normalize CPU RNG tensors on restore even when training runs on CUDA.
                state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                             best_loss=best_loss, train_loss=mean_loss, adadextrack=metadata,
                             rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                      cuda=[torch.cuda.get_rng_state(args.device)] if args.device.startswith("cuda") else []))
                if improved:
                    save_checkpoint(args.output_dir / "best.pth", state)
                save_checkpoint(args.output_dir / "last.pth", state)
                row = dict(epoch=epoch, action_mse=mean_loss, samples=rows, updates=steps, best_loss=best_loss)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                write_json(args.output_dir / "status.json", dict(status="running", **row))
                print("[BC-TRAIN] Epoch {}/{}: action MSE {:.6f} (best {:.6f})".format(
                    epoch, args.epochs, mean_loss, best_loss), flush=True)
        write_json(args.output_dir / "status.json", dict(status="complete", epoch=max(start_epoch, args.epochs), best_loss=best_loss))
    except BaseException as error:
        write_json(args.output_dir / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                        error=repr(error)))
        raise
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("examples", "unseen_ref", "unseen_obj"), default="examples")
    parser.add_argument("--latents-dir", type=Path, help="Check the representation already embedded in collected observations")
    parser.add_argument("--data-dir", type=Path, help="Default: outputs/bc_data/<dataset>")
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/bc/<dataset>")
    parser.add_argument("--device", default="cuda:0", help="One training device, default: cuda:0")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs, including restored epochs")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-mb", type=int, default=256, help="Maximum cached shard data on CPU (0 disables cache)")
    parser.add_argument("--network-config", type=Path, default=ROOT / "configs/bc_network.yaml")
    parser.add_argument("--resume", action="store_true", help="Restore last.pth, optimizer, normalization and RNG")
    args = parser.parse_args(argv)
    import math
    if (args.epochs < 1 or args.batch_size < 2 or args.seed < 0 or args.cache_mb < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0):
        parser.error("Use positive budgets/rates, batch-size >= 2, and nonnegative seed/cache size")
    args.data_dir = (args.data_dir or ROOT / "outputs/bc_data" / args.dataset).expanduser().resolve()
    args.output_dir = (args.output_dir or ROOT / "outputs/bc" / args.dataset).expanduser().resolve()
    args.network_config = args.network_config.expanduser().resolve()
    return args


def main(argv=None):
    return train(parse_args(argv))
