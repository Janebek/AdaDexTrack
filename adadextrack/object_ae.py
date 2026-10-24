"""Train, encode and decode object shapes without importing Isaac Gym."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from adadextrack.models.object_ae import PointAE, chamfer_l2

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def trimesh_module():
    try:
        import trimesh
        return trimesh
    except ImportError as error:
        raise RuntimeError('Install the optional AE dependencies: pip install -e ".[object-ae]"') from error


def mesh_paths(directory, include_obj=False):
    extensions = (".ply", ".obj") if include_obj else (".ply",)
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in extensions)
    if not paths:
        raise FileNotFoundError("No input meshes: " + str(directory))
    if len({p.stem.lower() for p in paths}) != len(paths):
        raise ValueError("Object names must be unique")
    return paths


def sample_points(path, num_points, seed):
    """Original surface/vertex sampling and unit-sphere normalization, explicit RNG."""
    trimesh = trimesh_module()
    obj = trimesh.load(str(path), process=False)
    if isinstance(obj, (list, tuple)):
        obj = trimesh.util.concatenate(obj)
    if isinstance(obj, trimesh.Trimesh) and len(obj.faces):
        points = trimesh.sample.sample_surface(obj, num_points, seed=int(seed))[0]
    else:
        if isinstance(obj, trimesh.Scene):
            if not obj.geometry:
                raise ValueError("Empty scene: " + str(path))
            obj = trimesh.util.concatenate(list(obj.geometry.values()))
        vertices = np.asarray(obj.vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("Invalid mesh vertices: " + str(path))
        choice = np.random.default_rng(seed).choice(len(vertices), num_points,
                                                   replace=len(vertices) < num_points)
        points = vertices[choice]
    points = points.astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("Non-finite input geometry: " + str(path))
    center = points.mean(axis=0, keepdims=True)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    scale = 1.0 if scale < 1e-8 else float(scale)
    return points / scale, center[0], scale


def split_indices(count, seed=42, val_split=.05):
    nval = int(count * val_split)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    return order[:count-nval], order[count-nval:]


class PointSetDataset(Dataset):
    def __init__(self, paths, num_points, seed):
        self.paths, self.num_points, self.seed, self.epoch = paths, num_points, seed, 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Stable across worker counts, resume and process restarts.
        seed = int(np.random.SeedSequence([self.seed, self.epoch, index]).generate_state(1)[0])
        return torch.from_numpy(sample_points(self.paths[index], self.num_points, seed)[0])


def load_model(path, device):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    model = PointAE(config["latent_dim"], config["num_points"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("AE weights contain non-finite values")
    return model.eval(), checkpoint


def save_checkpoint(path, state):
    temporary = Path(path).with_suffix(".pth.tmp")
    torch.save(state, str(temporary))
    temporary.replace(path)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    paths = mesh_paths(args.input, args.include_obj)
    records = [{"name": p.name, "sha256": digest(p)} for p in paths]
    train_ids, val_ids = split_indices(len(paths), args.seed, args.val_split)
    if len(train_ids) < args.batch_size:
        raise ValueError("Training samples must cover one full batch (drop_last=True)")
    dataset = PointSetDataset(paths, args.num_points, args.seed)
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(Subset(dataset, val_ids), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    config = dict(epochs=args.epochs, batch_size=args.batch_size, num_points=args.num_points,
                  latent_dim=args.latent_dim, lr=args.learning_rate, wd=args.weight_decay,
                  seed=args.seed, val_split=args.val_split, cdist_chunk=args.cdist_chunk)
    provenance = dict(schema_version=1, inputs=records, train=[paths[i].name for i in train_ids],
                      validation=[paths[i].name for i in val_ids], args=config,
                      sampling="SeedSequence([seed, epoch, sorted_file_index]); explicit Trimesh seed",
                      numpy=np.__version__, torch=torch.__version__, trimesh=trimesh_module().__version__)
    model = PointAE(args.latent_dim, args.num_points).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start, best = 1, float("inf")
    if args.resume:
        state = torch.load(str(args.resume), map_location="cpu", weights_only=False)
        if "provenance" in state and state["provenance"] != provenance:
            raise ValueError("AE resume inputs, sampling versions, split or training configuration differ")
        saved = state["args"]
        for key in config:
            if key in saved and saved[key] != config[key]:
                raise ValueError("AE resume setting differs: " + key)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optim"])
        scheduler.load_state_dict(state["sched"])
        start, best = int(state["epoch"]), float(state["best_val"])
        if state.get("rng"):
            random.setstate(state["rng"]["python"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"].cpu())
            if args.device.startswith("cuda") and state["rng"]["cuda"]:
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output_dir / name).exists() for name in ("last.pth", "best.pth")):
        raise FileExistsError("Use --resume or a new --output-dir")
    if args.resume and not (args.output_dir / "best.pth").exists():
        source_best = args.resume.parent / "best.pth"
        if not source_best.is_file() and args.resume.name in ("best.pth", "object_ae.pth"):
            source_best = args.resume
        if source_best.is_file():
            best_state = torch.load(str(source_best), map_location="cpu", weights_only=False)
            if (best_state["args"]["num_points"] != args.num_points
                    or best_state["args"]["latent_dim"] != args.latent_dim):
                raise ValueError("Sibling best.pth has an incompatible AE architecture")
            shutil.copyfile(str(source_best), str(args.output_dir / "best.pth"))
    write_json(args.output_dir / "provenance.json", provenance)
    if args.resume and start > args.epochs:
        save_checkpoint(args.output_dir / "last.pth", state)
        return
    for epoch in range(start, args.epochs + 1):
        dataset.epoch = epoch
        model.train()
        total = count = 0
        for points in train_loader:
            points = points.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = chamfer_l2(model(points)[0], points, args.cdist_chunk)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite AE training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(points)
            count += len(points)
        train_loss = total / count
        model.eval()
        total = count = 0
        with torch.no_grad():
            for points in val_loader:
                points = points.to(args.device)
                total += chamfer_l2(model(points)[0], points, args.cdist_chunk).item() * len(points)
                count += len(points)
        score = total / count if count else train_loss
        if not np.isfinite(score):
            raise ValueError("Non-finite AE validation loss")
        improved = score < best
        best = min(best, score)
        scheduler.step()
        state = dict(epoch=epoch+1, model=model.state_dict(), optim=optimizer.state_dict(),
                     sched=scheduler.state_dict(), best_val=best, args=config, provenance=provenance,
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []))
        save_checkpoint(args.output_dir / "last.pth", state)
        if improved:
            save_checkpoint(args.output_dir / "best.pth", state)
        print("[OBJECT-AE] Epoch {}/{} train={:.8g} validation={:.8g}".format(
            epoch, args.epochs, train_loss, score), flush=True)


def encode(args):
    model, checkpoint = load_model(args.checkpoint, args.device)
    paths = mesh_paths(args.input, args.include_obj)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.output_dir / (p.stem + "_latent.npy") for p in paths]
    if any(p.exists() for p in targets) or (args.output_dir / "encoding.json").exists():
        raise FileExistsError("Choose a new --output-dir; existing latents are never overwritten")
    records = []
    with torch.no_grad():
        for path, target in zip(paths, targets):
            points, center, scale = sample_points(path, checkpoint["args"]["num_points"], args.seed)
            latent = model.encoder(torch.from_numpy(points)[None].to(args.device))[0].cpu().numpy()
            if not np.isfinite(latent).all():
                raise ValueError("Non-finite encoded latent: " + str(path))
            np.save(str(target), latent.astype(np.float32))
            records.append(dict(input=str(path), input_sha256=digest(path), output=target.name,
                                output_sha256=digest(target), center=center.tolist(), scale=scale))
    write_json(args.output_dir / "encoding.json", dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
               seed=args.seed, num_points=checkpoint["args"]["num_points"], latent_dim=checkpoint["args"]["latent_dim"],
               sampling="Explicit seed for each mesh; surface sampling, then mean-center and max-radius normalization",
               coordinate_system="unit sphere of sampled points", numpy=np.__version__, torch=torch.__version__,
               trimesh=trimesh_module().__version__, device=args.device, files=records))
    print("[OBJECT-AE] Encoded {} objects: {}".format(len(records), args.output_dir))


def decode(args):
    model, _ = load_model(args.checkpoint, args.device)
    paths = sorted(args.latent_dir.glob("*_latent.npy"))
    if not paths:
        raise FileNotFoundError("No object latents: " + str(args.latent_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        target = args.output_dir / (path.name[:-len("_latent.npy")] + ".ply")
        if target.exists():
            raise FileExistsError(target)
        latent = np.load(str(path), allow_pickle=False)
        if latent.shape != (model.encoder.fc_mu.out_features,) or not np.isfinite(latent).all():
            raise ValueError("Invalid object latent: " + str(path))
        with torch.no_grad():
            points = model.decoder(torch.from_numpy(latent.astype(np.float32))[None].to(args.device))[0].cpu().numpy()
        trimesh_module().points.PointCloud(points).export(str(target))
        records.append(dict(input=path.name, input_sha256=digest(path), output=target.name, output_sha256=digest(target)))
    write_json(args.output_dir / "decoding.json", dict(checkpoint_sha256=digest(args.checkpoint),
               coordinate_system="normalized sampled-point unit sphere; original translation/scale not restored", files=records))
    print("[OBJECT-AE] Decoded {} objects: {}".format(len(records), args.output_dir))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--input", type=Path, default=ROOT / "data/object_ae/meshes")
    training.add_argument("--output-dir", type=Path, default=ROOT / "outputs/object_ae/train")
    training.add_argument("--epochs", type=int, default=1000)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--num-points", type=int, default=2048)
    training.add_argument("--latent-dim", type=int, default=64)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=0.)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--val-split", type=float, default=.05)
    training.add_argument("--cdist-chunk", type=int, default=0)
    training.add_argument("--resume", type=Path, help="Restore last.pth or the released original best.pth")
    encoding = commands.add_parser("encode")
    encoding.add_argument("--input", type=Path, default=ROOT / "data/object_ae/meshes")
    encoding.add_argument("--output-dir", type=Path, default=ROOT / "outputs/object_ae/latents")
    decoding = commands.add_parser("decode")
    decoding.add_argument("--latent-dir", type=Path, default=ROOT / "assets/latents")
    decoding.add_argument("--output-dir", type=Path, default=ROOT / "outputs/object_ae/reconstruction")
    for sub in (training, encoding, decoding):
        sub.add_argument("--device", default="cuda:0")
    for sub in (training, encoding):
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--include-obj", action="store_true")
    for sub in (encoding, decoding):
        sub.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/object_ae.pth")
    args = parser.parse_args(argv)
    if args.command == "train" and (min(args.epochs, args.batch_size, args.num_points, args.latent_dim) < 1
            or args.workers < 0 or args.cdist_chunk < 0 or not 0 <= args.val_split < 1
            or args.learning_rate <= 0 or args.weight_decay < 0):
        parser.error("Invalid training budget, split, optimizer or point count")
    if hasattr(args, "seed") and not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32)")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        {"train": train, "encode": encode, "decode": decode}[args.command](args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print("[OBJECT-AE] ERROR: {}".format(error), file=sys.stderr)
        return 1
    return 0
