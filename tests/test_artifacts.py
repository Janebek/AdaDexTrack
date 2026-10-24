import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from adadextrack.artifacts import (COMPONENTS, read_release, build_component, file_record, install_staged,
                                  unpack_verified, verify_component)


class ResourceBundlesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "source"
        self.output = self.base / "bundles"
        self.output.mkdir()
        for name, content in {
            "checkpoints/specialists/examples/group_322/best.pth": b"stamp specialist fixture",
            "checkpoints/specialists/examples/group_369/best.pth": b"toothpaste specialist fixture",
            "checkpoints/specialists/examples/group_435/best.pth": b"wineglass specialist fixture",
            "assets/urdf/objects/GRAB/cup/cup.obj": b"v 0 0 0\nf 1 1 1\n",
            "assets/xarm_leap/urdf/xarm_leap.urdf": b"<robot/>\n",
            "data/adadextrack_full/references/ref_0000.npy": b"reference fixture",
            "checkpoints/modulator_unseen_ref.pth": b"unseen-reference modulator fixture",
            "checkpoints/modulator_unseen_obj.pth": b"unseen-object modulator fixture",
            "checkpoints/object_ae.pth": b"AE checkpoint fixture",
            "data/object_ae/meshes/cup.ply": b"AE mesh fixture",
            "checkpoints/bc_tracker_unseen_ref.pth": b"checkpoint fixture",
            "checkpoints/bc_tracker_unseen_obj.pth": b"unseen-object checkpoint fixture",
        }.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_components_restore_exact_paths_and_bytes(self):
        staging = self.base / "staging"
        destination = self.base / "installed"
        for component in COMPONENTS:
            record = build_component(self.root, self.output, component)
            unpack_verified(self.output / record["filename"], component, record, staging)
        self.assertEqual(install_staged(staging, destination), 12)
        self.assertEqual(install_staged(staging, destination), 0)
        for source in self.root.rglob("*"):
            if source.is_file():
                self.assertEqual(source.read_bytes(), (destination / source.relative_to(self.root)).read_bytes())

    def test_legacy_release_without_unseen_object_tracker_is_readable(self):
        (self.root / "checkpoints/bc_tracker.pth").write_bytes(b"checkpoint fixture")
        components = {key: build_component(self.root, self.output, key)
                      for key in ("assets", "references", "bc_tracker")}
        manifest = self.output / "release.json"
        manifest.write_text(json.dumps(dict(schema_version=1, repo_id="Janebek/AdaDexTrack",
                                           components=components, metadata_files={})))
        self.assertEqual(set(read_release(manifest)["components"]), set(components))
        staging = self.base / "legacy-staging"
        record = read_release(manifest)["components"]["bc_tracker"]
        unpack_verified(self.output / record["filename"], "bc_tracker", record, staging)
        self.assertEqual((staging / "checkpoints/bc_tracker.pth").read_bytes(), b"checkpoint fixture")

    def test_reproducible_archives(self):
        first = build_component(self.root, self.output, "assets")
        other = self.base / "other"
        other.mkdir()
        second = build_component(self.root, other, "assets")
        self.assertEqual(first, second)

    def test_payload_corruption_is_rejected(self):
        record = build_component(self.root, self.output, "assets")
        archive = self.output / record["filename"]
        with archive.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_component(archive, "assets", record)

    def test_member_corruption_is_rejected(self):
        record = build_component(self.root, self.output, "assets")
        record["files"][next(iter(record["files"]))]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "member checksum"):
            verify_component(self.output / record["filename"], "assets", record)

    def test_unsafe_and_link_members_are_rejected(self):
        record = build_component(self.root, self.output, "assets")
        for name, linked in (("../escape", False), ("assets/../../escape", False),
                             ("assets/link", True)):
            with self.subTest(name=name):
                archive = self.base / "malformed.tar.gz"
                with tarfile.open(str(archive), "w:gz") as stream:
                    member = tarfile.TarInfo(name)
                    if linked:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../../escape"
                        stream.addfile(member)
                    else:
                        member.size = 1
                        stream.addfile(member, io.BytesIO(b"x"))
                bad = copy.deepcopy(record)
                bad.update(file_record(archive))
                with self.assertRaises(ValueError):
                    unpack_verified(archive, "assets", bad, self.base / "unsafe-output")
                self.assertFalse((self.base / "escape").exists())

    def test_incomplete_archive_is_rejected(self):
        record = build_component(self.root, self.output, "assets")
        archive = self.base / "empty.tar.gz"
        with tarfile.open(str(archive), "w:gz"):
            pass
        record.update(file_record(archive))
        with self.assertRaisesRegex(ValueError, "Incomplete archive"):
            verify_component(archive, "assets", record)

    def test_conflict_is_detected_before_any_file_is_installed(self):
        staging, destination = self.base / "stage", self.base / "target"
        staging.mkdir()
        destination.mkdir()
        (staging / "a-new").write_bytes(b"new")
        (staging / "z-existing").write_bytes(b"replacement")
        (destination / "z-existing").write_bytes(b"original")
        with self.assertRaises(ValueError):
            install_staged(staging, destination)
        self.assertFalse((destination / "a-new").exists())
        self.assertEqual((destination / "z-existing").read_bytes(), b"original")

    def test_source_and_destination_symlinks_are_rejected(self):
        (self.root / "assets/link").symlink_to(self.root / "checkpoints/bc_tracker_unseen_ref.pth")
        with self.assertRaisesRegex(ValueError, "Symbolic"):
            build_component(self.root, self.output, "assets")
        stage, target, outside = self.base / "stage", self.base / "target", self.base / "outside"
        (stage / "assets").mkdir(parents=True)
        (stage / "assets/file").write_bytes(b"new")
        target.mkdir()
        outside.mkdir()
        (target / "assets").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            install_staged(stage, target)
        self.assertFalse((outside / "file").exists())

    def test_downloader_handles_hub_snapshot_symlinks_and_pins_revision(self):
        records = {key: build_component(self.root, self.output, key)
                   for key in COMPONENTS}
        manifest = self.output / "release.json"
        manifest.write_text(json.dumps(dict(schema_version=1, repo_id="Janebek/AdaDexTrack",
                                           components=records, metadata_files={})))
        snapshot = self.base / "snapshot"
        snapshot.mkdir()
        for path in self.output.iterdir():
            (snapshot / path.name).symlink_to(path)
        hub = ModuleType("huggingface_hub")
        hub.HfApi = Mock()
        revision = "a" * 40
        hub.HfApi.return_value.repo_info.return_value = SimpleNamespace(sha=revision)
        hub.hf_hub_download = Mock(side_effect=lambda filename, **kwargs: str(snapshot / filename))
        entrypoint = Path(__file__).resolve().parents[1] / "scripts/download_resources.py"
        spec = importlib.util.spec_from_file_location("resource_download_entrypoint", entrypoint)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        destination = self.base / "downloaded"
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(
                sys, "argv", [str(entrypoint), "--root", str(destination)]):
            module.main()
        self.assertEqual((destination / "checkpoints/bc_tracker_unseen_ref.pth").read_bytes(), b"checkpoint fixture")
        self.assertEqual((destination / "checkpoints/bc_tracker_unseen_obj.pth").read_bytes(),
                         b"unseen-object checkpoint fixture")
        self.assertEqual(len(hub.hf_hub_download.call_args_list), 7)
        for call in hub.hf_hub_download.call_args_list:
            self.assertEqual(call.kwargs["revision"], revision)
        self.assertEqual((destination / "checkpoints/modulator_unseen_ref.pth").read_bytes(),
                         b"unseen-reference modulator fixture")
        self.assertEqual((destination / "checkpoints/modulator_unseen_obj.pth").read_bytes(),
                         b"unseen-object modulator fixture")
        self.assertFalse((destination / "checkpoints/specialists").exists())
        # Specialists are opt-in and restore all three group-specific checkpoints.
        specialist_destination = self.base / "specialist-download"
        hub.hf_hub_download.reset_mock()
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(
                sys, "argv", [str(entrypoint), "--root", str(specialist_destination),
                              "--components", "example_specialists"]):
            module.main()
        self.assertEqual(len(hub.hf_hub_download.call_args_list), 2)
        for group_id in (322, 369, 435):
            relative = "checkpoints/specialists/examples/group_{}/best.pth".format(group_id)
            self.assertEqual((specialist_destination / relative).read_bytes(),
                             (self.root / relative).read_bytes())
        self.assertFalse((specialist_destination / "assets").exists())
        # A pinned older release still defaults to the components it actually has.
        old_records = {key: record for key, record in records.items() if not key.startswith("modulator_")}
        manifest.write_text(json.dumps(dict(schema_version=1, repo_id="Janebek/AdaDexTrack",
                                           components=old_records, metadata_files={})))
        old_destination = self.base / "older-download"
        hub.hf_hub_download.reset_mock()
        with patch.dict(sys.modules, {"huggingface_hub": hub}), patch.object(
                sys, "argv", [str(entrypoint), "--root", str(old_destination)]):
            module.main()
        self.assertEqual(len(hub.hf_hub_download.call_args_list), 5)
        self.assertTrue((old_destination / "checkpoints/bc_tracker_unseen_ref.pth").is_file())
        self.assertFalse((old_destination / "checkpoints/modulator_unseen_ref.pth").exists())


if __name__ == "__main__":
    unittest.main()
