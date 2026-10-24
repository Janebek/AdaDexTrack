"""Shared semantic-group selection for replay and specialist training."""

from pathlib import Path

from adadextrack.data.dataset import read_manifest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/adadextrack_full"
SOURCES = {
    "examples": DATA / "examples/specialists.csv",
    "full": DATA / "manifest.csv",
    "unseen_ref": DATA / "splits/unseen_trajectory_train.csv",
    "unseen_obj": DATA / "splits/unseen_object_train.csv",
}


def partition_groups(entries):
    if not entries:
        raise ValueError("Empty semantic-group selection")
    grouped, seen = {}, set()
    for entry in entries:
        if entry.reference_id in seen:
            raise ValueError("Duplicate reference ID: " + entry.reference_id)
        seen.add(entry.reference_id)
        grouped.setdefault(entry.group_id, []).append(entry)
    groups = [grouped[key] for key in sorted(grouped, key=int)]
    for group in groups:
        if len({entry.object_name for entry in group}) != 1:
            raise ValueError("A semantic group must contain exactly one object")
    return groups


def source_manifest(dataset=None, manifest=None):
    if dataset is not None and manifest is not None:
        raise ValueError("Use either --dataset or --manifest")
    return Path(manifest).expanduser().resolve() if manifest is not None else SOURCES[dataset or "examples"]


def select_groups(manifest, group_ids=None, all_groups=False, reference_ids=None, verify_hashes=False):
    entries = read_manifest(manifest, reference_ids=reference_ids, verify_hashes=verify_hashes)
    groups = partition_groups(entries)
    if group_ids is not None:
        wanted = list(map(str, group_ids))
        if not wanted or len(set(wanted)) != len(wanted):
            raise ValueError("Group IDs must be nonempty and unique")
        available = {group[0].group_id for group in groups}
        if not set(wanted).issubset(available):
            raise ValueError("Group ID is absent from the selected data source")
        groups = [group for group in groups if group[0].group_id in wanted]
    elif reference_ids is None and not all_groups:
        groups = groups[:1]
    return groups
