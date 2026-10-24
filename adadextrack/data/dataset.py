"""Read a portable AdaDexTrack reference manifest.

IDs are local to the dataset. Reference paths are relative to the dataset root
(the ancestor containing dataset.json), including in per-group manifests.
This module does not import Isaac Gym or torch until task tensors are requested.
"""

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class DatasetReference:
    reference_id: str
    group_id: str
    object_name: str
    path: str
    num_frames: int
    description: str
    sha256: str


def read_manifest(path, reference_ids=None, verify_hashes=False):
    path = Path(path).expanduser().resolve()
    root = next((parent for parent in path.parents if (parent / "dataset.json").is_file()), None)
    if root is None:
        raise FileNotFoundError("No dataset.json ancestor for manifest {}".format(path))
    metadata = json.loads((root / "dataset.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported reference dataset schema")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"reference_id", "group_id", "object_name", "reference_file", "num_frames", "description", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Incomplete reference manifest columns: {}".format(path))
        rows = list(reader)
    entries, used_ids, used_paths = [], set(), set()
    for row in rows:
        ref_id, group_id = row["reference_id"], row["group_id"]
        if not all(re.fullmatch(r"0|[1-9]\d*", value) for value in (ref_id, group_id)):
            raise ValueError("Reference/group IDs must be canonical nonnegative integers")
        relative = Path(row["reference_file"])
        filename = (root / relative).resolve()
        if relative.is_absolute() or root not in filename.parents or not filename.is_file():
            raise ValueError("Reference must be a file inside the dataset: {}".format(relative))
        if ref_id in used_ids or filename in used_paths:
            raise ValueError("Duplicate reference ID or path: {}".format(ref_id))
        used_ids.add(ref_id)
        used_paths.add(filename)
        obj = row["object_name"]
        if not obj or Path(obj).name != obj or obj in (".", ".."):
            raise ValueError("Invalid object name: {}".format(obj))
        if int(row["num_frames"]) != 300 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid frame count or SHA256 for reference {}".format(ref_id))
        entries.append(DatasetReference(ref_id, group_id, obj, str(filename), 300, row["description"], row["sha256"]))
    if not entries:
        raise ValueError("Empty reference manifest: {}".format(path))
    if reference_ids is not None:
        selected = [str(value) for value in reference_ids]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Selected reference IDs must be nonempty and unique")
        by_id = {entry.reference_id: entry for entry in entries}
        if not set(selected).issubset(by_id):
            raise ValueError("Selected reference IDs are absent from the manifest")
        entries = [by_id[ref_id] for ref_id in selected]
    if verify_hashes:
        for entry in entries:
            if hashlib.sha256(Path(entry.path).read_bytes()).hexdigest() != entry.sha256:
                raise ValueError("Reference checksum mismatch: {}".format(entry.reference_id))
    return entries


def load_task_reference_tensors(entries, device):
    """Produce the existing AllAll10 reference buffers with identical ordering."""
    import torch
    import torch.nn.functional as F

    positions, rotations, joints = [], [], []
    inverse = np.argsort(list(range(10)) + list(range(14, 22)) + [10, 11, 12, 13])
    for entry in entries:
        data = np.load(entry.path, allow_pickle=True).item()
        arrays = [np.asarray(data[key]) for key in ("object_transl", "object_rot_quat", "robot_delta_states_weights_np")]
        for array, width in zip(arrays, (3, 4, 22)):
            if array.shape != (300, width) or not np.isfinite(array).all():
                raise ValueError("Invalid reference arrays: {}".format(entry.path))
        if np.any(np.linalg.norm(arrays[1], axis=-1) < 1e-8):
            raise ValueError("Invalid reference quaternion: {}".format(entry.path))
        positions.append(arrays[0])
        rotations.append(arrays[1])
        joints.append(arrays[2][:, inverse])
    pos = torch.as_tensor(np.stack(positions), dtype=torch.float32, device=device)
    rot = F.normalize(torch.as_tensor(np.stack(rotations), dtype=torch.float32, device=device), p=2, dim=-1, eps=1e-8)
    hand = torch.as_tensor(np.stack(joints), dtype=torch.float32, device=device)
    lengths = torch.full((len(entries),), 300, device=device, dtype=torch.long)
    mask = torch.ones((len(entries), 300), device=device, dtype=torch.bool)
    return pos, rot, hand, lengths, mask, [Path(entry.path).name for entry in entries], [entry.path for entry in entries]
