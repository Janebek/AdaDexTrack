"""Prepare, verify and publish the explicitly selected AdaDexTrack Hub resources."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adadextrack.artifacts import COMPONENTS, file_record, verify_release, build_component


def prepare(root, output, repo_id):
    if output.exists():
        raise FileExistsError("Use --verify-only or an unused --output-dir: {}".format(output))
    dataset = json.loads((root / "data/adadextrack_full/dataset.json").read_text())
    trackers = json.loads((root / "checkpoints/checkpoints.json").read_text())
    for name in ("bc_tracker_unseen_ref", "bc_tracker_unseen_obj", "modulator_unseen_ref", "modulator_unseen_obj"):
        tracker = trackers[name]
        if file_record(root / "checkpoints" / tracker["file"])["sha256"] != tracker["sha256"]:
            raise ValueError("{} differs from checkpoints/checkpoints.json".format(name))
    examples = trackers["small_scale_specialists"]
    if {item["group_id"] for item in examples["models"]} != set(examples["group_ids"]):
        raise ValueError("Example specialist registry does not cover all three groups")
    for specialist in examples["models"]:
        expected = {key: specialist[key] for key in ("bytes", "sha256")}
        if file_record(root / "checkpoints" / specialist["file"]) != expected:
            raise ValueError("Example specialist differs from its registered checksum")
    ae = trackers["object_ae"]
    if file_record(root / "checkpoints" / ae["file"])["sha256"] != ae["sha256"]:
        raise ValueError("Object AE weights differ from their registered checksum")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-release-", dir=str(output.parent)) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        components = {}
        for component in COMPONENTS:
            print("[HF] Packing {}".format(component), flush=True)
            components[component] = build_component(root, staging, component)
        reference_files = components["references"]["files"]
        reference_count = sum(name.startswith("data/adadextrack_full/references/")
                              and name.endswith(".npy") for name in reference_files)
        if reference_count != dataset["references"]:
            raise ValueError("Reference package does not cover the complete dataset")
        metadata = {
            "README.md": root / "docs/huggingface_dataset_card.md",
            "RESOURCE_LICENSES.md": root / "docs/resource_licenses.md",
        }
        for path in sorted((root / "third_party").glob("*LICENSE*")):
            metadata["notices/" + path.name] = path
        metadata_records = {}
        for name, source in metadata.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(source), str(target))
            metadata_records[name] = file_record(target)
        release = dict(schema_version=1, repo_id=repo_id, repo_type="dataset",
                       created_utc=datetime.now(timezone.utc).isoformat(),
                       dataset={key: dataset[key] for key in
                                ("name", "references", "objects", "semantic_groups", "frames_per_reference", "splits")},
                       components=components, metadata_files=metadata_records)
        (staging / "release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
        verify_release(staging)
        staging.rename(output)
    return release


def publish(output, repo_id):
    # Load the official client only for publication: packing works without Hub login.
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    release = verify_release(output)
    if release["repo_id"] != repo_id:
        raise ValueError("Package repository and --repo-id differ")
    api = HfApi()
    try:
        api.whoami()
    except Exception as error:
        if getattr(getattr(error, "response", None), "status_code", None) == 401:
            raise RuntimeError("Hugging Face login required. Run huggingface-cli login in your terminal; "
                               "do not put a token in a command or chat.") from error
        raise RuntimeError("Unable to verify the Hugging Face account; check the connection/client settings") from error
    parent_commit = None
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        parent_commit = info.sha
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    allow = ["release.json"] + list(release["metadata_files"])
    allow.extend(record["filename"] for record in release["components"].values())
    result = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(output),
                               allow_patterns=allow, parent_commit=parent_commit,
                               commit_message="Release three pretrained example semantic specialists")
    report = dict(repo_id=repo_id, repo_type="dataset", commit_url=str(result),
                  revision=getattr(result, "oid", None),
                  uploaded_files=allow, uploaded_utc=datetime.now(timezone.utc).isoformat())
    (output.parent / (output.name + "_upload.json")).write_text(json.dumps(report, indent=2) + "\n")
    print("[HF] Published: {}".format(result), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Janebek/AdaDexTrack")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/huggingface/release_v6")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Build and verify bundles without networking")
    mode.add_argument("--verify-only", action="store_true", help="Verify existing bundles without networking")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if args.verify_only:
        release = verify_release(output)
    elif output.exists() and not args.prepare_only:
        release = verify_release(output)
    else:
        release = prepare(ROOT, output, args.repo_id)
    for component, record in release["components"].items():
        print("[HF] {}: {} files, {:.2f} MB, SHA256 {}".format(
            component, len(record["files"]), record["bytes"] / 1e6, record["sha256"]), flush=True)
    if not args.prepare_only and not args.verify_only:
        publish(output, args.repo_id)
    else:
        print("[HF] Verified local release: {}".format(output), flush=True)


if __name__ == "__main__":
    main()
