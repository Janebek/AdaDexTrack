"""Download versioned Hub resources, verify checksums and restore runtime paths."""

import argparse
import inspect
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adadextrack.artifacts import COMPONENTS, read_release, unpack_verified, install_staged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Janebek/AdaDexTrack")
    release_config = ROOT / "configs/artifact_release.json"
    pinned = json.loads(release_config.read_text()).get("revision") if release_config.is_file() else None
    parser.add_argument("--revision", default=pinned or "main",
                        help="Hub commit, tag or branch (default: pinned project revision, otherwise main)")
    parser.add_argument("--components", nargs="+", choices=tuple(COMPONENTS),
                        help="Components to install (default: assets, references, both BC trackers and both modulators; example specialists and AE components are optional)")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    # Pin all files to one immutable revision, including when main changes mid-download.
    revision = HfApi().repo_info(repo_id=args.repo_id, repo_type="dataset", revision=args.revision).sha
    options = dict(repo_id=args.repo_id, repo_type="dataset", revision=revision)
    if args.cache_dir:
        options["cache_dir"] = str(args.cache_dir.expanduser())
    if "resume_download" in inspect.signature(hf_hub_download).parameters:
        options["resume_download"] = True
    manifest = hf_hub_download(filename="release.json", **options)
    release = read_release(manifest)
    if release["repo_id"] != args.repo_id:
        raise ValueError("Downloaded manifest repository does not match --repo-id")
    components = args.components if args.components is not None else [name for name in release["components"] if name not in ("object_ae", "object_ae_data", "example_specialists")]
    missing = set(components) - set(release["components"])
    if missing:
        raise ValueError("Selected Hub revision does not contain: {}".format(", ".join(sorted(missing))))
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-install-", dir=str(root)) as temporary:
        staging = Path(temporary)
        for component in dict.fromkeys(components):
            record = release["components"][component]
            print("[HF] Downloading and verifying {}".format(component), flush=True)
            path = hf_hub_download(filename=record["filename"], **options)
            # Hub snapshot entries are normally symlinks into its immutable blob cache.
            unpack_verified(Path(path).resolve(), component, record, staging)
        installed = install_staged(staging, root)
    receipt = dict(repo_id=args.repo_id, revision=revision, components=components,
                   installed_new_files=installed)
    receipt_dir = root / "outputs/resources"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "download.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("[HF] Installed {} new files at {} (revision {})".format(installed, root, revision), flush=True)


if __name__ == "__main__":
    main()
