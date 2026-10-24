"""Checksummed resource bundles shared by the Hub publisher and downloader."""

import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


COMPONENTS = {
    "example_specialists": ("example_specialists.tar.gz", "checkpoints/specialists/examples", "tar.gz"),
    "object_ae": ("object_ae.pth", "checkpoints/object_ae.pth", "file"),
    "object_ae_data": ("object_ae_data.tar.gz", "data/object_ae", "tar.gz"),
    "modulator_unseen_ref": ("modulator_unseen_ref.pth", "checkpoints/modulator_unseen_ref.pth", "file"),
    "modulator_unseen_obj": ("modulator_unseen_obj.pth", "checkpoints/modulator_unseen_obj.pth", "file"),
    "assets": ("assets.tar.gz", "assets", "tar.gz"),
    "references": ("references.tar.gz", "data/adadextrack_full", "tar.gz"),
    "bc_tracker_unseen_ref": ("bc_tracker_unseen_ref.pth", "checkpoints/bc_tracker_unseen_ref.pth", "file"),
    "bc_tracker_unseen_obj": ("bc_tracker_unseen_obj.pth", "checkpoints/bc_tracker_unseen_obj.pth", "file"),
}


# Layout used by immutable releases published before the explicit tracker names.
LEGACY_COMPONENTS = {
    "bc_tracker": ("bc_tracker.pth", "checkpoints/bc_tracker.pth", "file"),
}


def component_layout(component):
    return (COMPONENTS if component in COMPONENTS else LEGACY_COMPONENTS)[component]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name, prefix):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or path.is_absolute() or ".." in path.parts
            or "\\" in name or ":" in name or str(path) != name
            or not (name == prefix or name.startswith(prefix + "/"))):
        raise ValueError("Invalid resource path: {}".format(name))
    return name


def file_record(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def check_file(path, record):
    if not path.is_file() or path.is_symlink():
        raise ValueError("Missing regular file: {}".format(path))
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError("Resource checksum mismatch: {}".format(path))


def build_component(root, output, component):
    root, output = Path(root).resolve(), Path(output)
    filename, prefix, kind = component_layout(component)
    source = root / prefix
    if source.is_symlink() or not source.exists():
        raise ValueError("Missing resource or symbolic link: {}".format(source))
    candidates = [source] if kind == "file" else sorted(source.rglob("*"))
    files = []
    for path in candidates:
        if path.is_symlink():
            raise ValueError("Symbolic links are not bundled: {}".format(path))
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Non-regular resource: {}".format(path))
        if "__pycache__" not in path.parts and path.suffix != ".pyc":
            files.append(path)
    if not files:
        raise ValueError("Empty resource component: {}".format(component))
    inventory = {safe_name(path.relative_to(root).as_posix(), prefix): file_record(path)
                 for path in files}
    target = output / filename
    if target.exists():
        raise FileExistsError(str(target))
    if kind == "file":
        shutil.copyfile(str(source), str(target))
    else:
        with target.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=4) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for path in files:
                        name = path.relative_to(root).as_posix()
                        info = tarfile.TarInfo(name)
                        info.size = inventory[name]["bytes"]
                        info.mode = 0o644
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
    record = dict(file_record(target), filename=filename, format=kind, files=inventory)
    verify_component(target, component, record)
    return record


def validate_record(component, record):
    filename, prefix, kind = component_layout(component)
    if record["filename"] != filename or record["format"] != kind or not record["files"]:
        raise ValueError("Unexpected component layout: {}".format(component))
    for name in record["files"]:
        safe_name(name, prefix)
    if kind == "file" and set(record["files"]) != {prefix}:
        raise ValueError("Unexpected single-file component")


def unpack_verified(path, component, record, destination=None):
    """Verify every member; optionally write only regular files into a staging dir."""
    validate_record(component, record)
    check_file(path, record)
    destination = Path(destination).resolve() if destination is not None else None
    if record["format"] == "file":
        name, expected = next(iter(record["files"].items()))
        check_file(path, expected)
        if destination is not None:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
        return
    seen = set()
    with tarfile.open(str(path), mode="r:gz") as archive:
        for member in archive:
            name = member.name
            safe_name(name, component_layout(component)[1])
            if not member.isfile() or name not in record["files"] or name in seen:
                raise ValueError("Unexpected archive member: {}".format(name))
            seen.add(name)
            expected = record["files"][name]
            if member.size != expected["bytes"]:
                raise ValueError("Archive member size mismatch: {}".format(name))
            source = archive.extractfile(member)
            digest = hashlib.sha256()
            target = None
            if destination is not None:
                target_path = destination / name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target = target_path.open("xb")
            try:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    if target is not None:
                        target.write(chunk)
            finally:
                source.close()
                if target is not None:
                    target.close()
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("Archive member checksum mismatch: {}".format(name))
    if seen != set(record["files"]):
        raise ValueError("Incomplete archive: {}".format(component))


def verify_component(path, component, record):
    unpack_verified(path, component, record)


def read_release(path):
    release = json.loads(Path(path).read_text(encoding="utf-8"))
    # Keep the original immutable releases readable under their original names.
    components = set(release["components"])
    ref_trackers = components & {"bc_tracker", "bc_tracker_unseen_ref"}
    if (release.get("schema_version") != 1
            or not {"assets", "references"}.issubset(components)
            or len(ref_trackers) != 1
            or not components.issubset(set(COMPONENTS) | set(LEGACY_COMPONENTS))):
        raise ValueError("Unsupported artifact release manifest")
    for component, record in release["components"].items():
        validate_record(component, record)
    for name in release.get("metadata_files", {}):
        candidate = PurePosixPath(name)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in name or str(candidate) != name:
            raise ValueError("Invalid release metadata path")
    return release


def verify_release(folder):
    folder = Path(folder)
    release = read_release(folder / "release.json")
    for name, record in release["metadata_files"].items():
        check_file(folder / name, record)
    for component, record in release["components"].items():
        verify_component(folder / record["filename"], component, record)
    return release


def install_staged(staging, root):
    """Reject all conflicts before installing any new file; keep identical files."""
    staging, root = Path(staging).resolve(), Path(root).resolve()
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    pending = []
    for path in files:
        target = root / path.relative_to(staging)
        for parent in (target,) + tuple(target.parents):
            if parent == root:
                break
            if parent.is_symlink() or (parent.exists() and parent != target and not parent.is_dir()):
                raise ValueError("Conflicting resource path: {}".format(parent))
        if target.exists():
            check_file(target, file_record(path))
        else:
            pending.append((path, target))
    for path, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output)
    return len(pending)
