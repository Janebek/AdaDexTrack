"""Record every reference in an example semantic group with matched cameras."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adadextrack.data.groups import source_manifest, select_groups
from adadextrack.visualization import GroupRecorder, CAMERA_POSITION, CAMERA_TARGET, CAMERA_FOV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "specialist"), required=True)
    parser.add_argument("--group-id", choices=(322, 369, 435), type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/example_visualizations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-position", nargs=3, type=float, default=CAMERA_POSITION)
    parser.add_argument("--camera-target", nargs=3, type=float, default=CAMERA_TARGET)
    parser.add_argument("--fov", type=float, default=CAMERA_FOV)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args._worker:
        for group in ([args.group_id] if args.group_id is not None else [322, 369, 435]):
            command = [sys.executable, "-u", str(Path(__file__).resolve())] + sys.argv[1:]
            if args.group_id is None:
                command += ["--group-id", str(group)]
            subprocess.run(command + ["--_worker"], check=True, cwd=str(ROOT))
        return 0
    # The extension must precede torch, including the evaluator's CPU preflight.
    import isaacgym
    manifest = source_manifest("examples")
    entries = select_groups(manifest, group_ids=[args.group_id], verify_hashes=True)[0]
    directory = args.output_dir.resolve() / args.mode / "group_{:03d}".format(args.group_id)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "{}_group_{}.mp4".format(args.mode, args.group_id)

    def recorder_factory(gym, sim, envs):
        return GroupRecorder(gym, sim, envs, entries, output, args.mode,
                             position=args.camera_position, target=args.camera_target,
                             fov=args.fov, width=args.width, height=args.height)

    if args.mode == "replay":
        from adadextrack.replay import replay
        selection = argparse.Namespace(mode="group", device=args.device, headless=True,
            overview=False, snapshot=None, loops=1, fps=12, output=directory / "states.npz")
        replay(selection, entries=entries, recorder_factory=recorder_factory)
    else:
        from adadextrack.inference import parse_args, preflight, worker, write_json
        checkpoint = ROOT / "checkpoints/specialists/examples/group_{:03d}/best.pth".format(args.group_id)
        evaluation = parse_args(["--ref-csv", str(manifest), "--group-id", str(args.group_id),
            "--checkpoint", str(checkpoint), "--device", args.device, "--seed", str(args.seed)])
        refs, versions = preflight(evaluation)
        config = {"checkpoint": str(checkpoint), "device": args.device, "seed": args.seed,
            "reference_manifest": str(manifest), "latents_dir": str(evaluation.latents_dir),
            "latents": versions["latents"], "render": False, "policy_type": "tracker"}
        path = directory / "manifest.json"
        write_json(path, {"settings": config, "batch_index": 0,
                          "references": [asdict(ref) for ref in refs]})
        worker(path, recorder_factory=recorder_factory)
    print("[VIDEO] " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
