#!/usr/bin/env python3
"""Record synchronized, single-reference physics comparisons of BC with/without modulation."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_REFERENCES = (1566, 258, 973, 478, 674, 2579)
CAMERA_POSITION = (1.18, -0.93, 1.25)
CAMERA_TARGET = (0.45, 0.30, 0.50)
CAMERA_FOV = 48.0
SIZE = 640
PANEL_GAP = 48
PANEL_MARGIN = 36
DISPLAY_WIDTH = 350
CONDITIONS = ("with_modulator", "without_modulator")
LABELS = {"with_modulator": "w/ Modulator", "without_modulator": "w/o Modulator"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoder_environment():
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    return env


def evaluate(args):
    # Import before the evaluator's torch-based checkpoint preflight.
    import isaacgym
    from adadextrack import inference
    from adadextrack.data.dataset import read_manifest
    from adadextrack.visualization import GroupRecorder
    from scripts.infer_modulator import parse_args

    condition = args._condition
    directory = args.output_dir / ("ref_{:04d}".format(args.reference_id[0])) / condition
    directory.mkdir(parents=True, exist_ok=True)
    policy_args = parse_args(["--dataset", args.dataset, "--reference-id", str(args.reference_id[0]),
                              "--device", args.device, "--seed", str(args.seed)])
    if condition == "without_modulator":
        policy_args.policy_type = "tracker"
        policy_args.checkpoint = policy_args.tracker_checkpoint
    refs, versions = inference.preflight(policy_args)
    entries = read_manifest(policy_args.ref_csv, reference_ids=args.reference_id, verify_hashes=True)
    if len(refs) != 1 or len(entries) != 1:
        raise ValueError("Each comparison worker must load exactly one reference")
    settings = dict(checkpoint=str(policy_args.checkpoint),
        tracker_checkpoint=str(policy_args.tracker_checkpoint), policy_type=policy_args.policy_type,
        device=args.device, seed=args.seed, reference_manifest=str(policy_args.ref_csv),
        latents_dir=str(policy_args.latents_dir), latents=versions["latents"], render=False)
    manifest = directory / "manifest.json"
    inference.write_json(manifest, dict(settings=settings, batch_index=0,
                                        references=[asdict(ref) for ref in refs]))
    title = "Reference {} | {}".format(entries[0].reference_id, entries[0].object_name)

    def recorder(gym, sim, envs):
        return GroupRecorder(gym, sim, envs, entries, directory / "recording.mp4", condition,
            position=args.camera_position, target=args.camera_target, fov=args.fov,
            width=SIZE, height=SIZE, title=title, subtitle="Physics simulation",
            panel_labels=[LABELS[condition]])

    inference.worker(manifest, recorder_factory=recorder)
    inference.write_json(directory / "provenance.json", dict(
        checkpoint=str(policy_args.checkpoint.relative_to(ROOT)),
        checkpoint_sha256=versions["checkpoint_sha256"],
        tracker_checkpoint=str(policy_args.tracker_checkpoint.relative_to(ROOT)),
        tracker_sha256=sha256(policy_args.tracker_checkpoint), latents=versions["latents"]))


def compose(args, reference_id):
    from adadextrack.data.dataset import read_manifest
    from adadextrack.inference import write_json
    from scripts.infer_modulator import parse_args

    base = args.output_dir / ("ref_{:04d}".format(reference_id))
    dirs = [base / condition for condition in CONDITIONS]
    records = [json.loads((d / "recording.json").read_text()) for d in dirs]
    fields = ("reference_ids", "camera_position", "camera_target", "horizontal_fov",
              "camera_size", "video_size", "fps", "frames", "status")
    if any(records[0][key] != records[1][key] for key in fields):
        raise ValueError("Comparison cameras, references or frame counts differ")
    if records[0]["frames"] != 300 or records[0]["status"] != "complete":
        raise ValueError("Each comparison requires two complete 300-frame recordings")
    policy_args = parse_args(["--dataset", args.dataset, "--reference-id", str(reference_id)])
    entry, = read_manifest(policy_args.ref_csv, reference_ids=[reference_id], verify_hashes=True)
    if not entry.object_name.replace("_", "").isalnum():
        raise ValueError("Object name cannot be rendered as a video label")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install FFmpeg to compose comparisons")
    title = "Reference {} | {}".format(reference_id, entry.object_name)
    left_x = PANEL_MARGIN
    right_x = left_x + SIZE + PANEL_GAP
    video_width = right_x + SIZE + PANEL_MARGIN
    camera_y = 132
    video_height = camera_y + SIZE + 8
    # Retain the camera resolution and use readable labels for 350 px grid previews.
    filters = [
        "[0:v]crop=640:640:8:102,setpts=PTS-STARTPTS,pad={}:{}:{}:{}:color=0xebf0f6[left]".format(video_width, video_height, left_x, camera_y),
        "[1:v]crop=640:640:8:102,setpts=PTS-STARTPTS[right]",
        "[left][right]overlay={}:{},".format(right_x, camera_y) + ",".join([
            "drawbox=x=0:y=0:w=iw:h=64:color=0x182637:t=fill",
            "drawbox=x={}:y=72:w=640:h=60:color=white:t=fill".format(left_x),
            "drawbox=x={}:y=72:w=640:h=60:color=white:t=fill".format(right_x),
            "drawbox=x={}:y=72:w=4:h=60:color=0x40a080:t=fill".format(left_x),
            "drawbox=x={}:y=72:w=4:h=60:color=0x8995a3:t=fill".format(right_x),
            "drawtext=font=DejaVuSans-Bold:text='{}':x=16:y=10:fontsize=40:fontcolor=white".format(title),
            "drawtext=font=DejaVuSans-Bold:text='w/ Modulator':x={}:y=82:fontsize=40:fontcolor=0x2c3d50".format(left_x + 12),
            "drawtext=font=DejaVuSans-Bold:text='w/o Modulator':x={}:y=82:fontsize=40:fontcolor=0x2c3d50".format(right_x + 12),
        ]) + "[video]",
    ]
    video = base / "comparison.mp4"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n",
        "-i", str(dirs[0] / "recording.mp4"), "-i", str(dirs[1] / "recording.mp4"),
        "-filter_complex_threads", "1", "-filter_complex", ";".join(filters), "-map", "[video]",
        "-frames:v", "300", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
        "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
        check=True, env=encoder_environment())
    preview = base / "comparison.webp"
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
        "-i", str(video), "-c:v", "libwebp_anim",
        "-quality", "88", "-loop", "0", "-an", "-vsync", "0", str(preview)],
        check=True, env=encoder_environment())
    info = dict(reference_id=entry.reference_id, object_name=entry.object_name,
        description=entry.description, reference_sha256=entry.sha256, dataset=args.dataset,
        split="test", seed=args.seed, frames=300, fps=12, duration_seconds=25,
        camera_position=records[0]["camera_position"], camera_target=records[0]["camera_target"],
        horizontal_fov=records[0]["horizontal_fov"], camera_size=[SIZE, SIZE], video_size=[video_width, video_height], panel_gap_px=PANEL_GAP,
        panel_margin_px=PANEL_MARGIN,
        display_width_px=DISPLAY_WIDTH,
        display_panel_width_px=SIZE * DISPLAY_WIDTH // video_width,
        display_panel_gap_px=PANEL_GAP * DISPLAY_WIDTH // video_width,
        display_panel_margin_px=PANEL_MARGIN * DISPLAY_WIDTH // video_width,
        initial_pose_perturbation=False, domain_randomization=False, lag=False,
        observation_noise="Existing task Gaussian noise retained (std 0.002).",
        video=video.name, video_sha256=sha256(video), preview=preview.name, preview_sha256=sha256(preview),
        left="with_modulator", right="without_modulator")
    for condition, directory in zip(CONDITIONS, dirs):
        metrics, = json.loads((directory / "metrics.json").read_text())
        info[condition] = dict(metrics=metrics, **json.loads((directory / "provenance.json").read_text()))
    write_json(base / "comparison.json", info)
    print("[COMPARISON] Reference {}: Completion {:.1%} / {:.1%}. {}".format(reference_id,
        info["with_modulator"]["metrics"]["completion_rate"],
        info["without_modulator"]["metrics"]["completion_rate"], video), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("unseen_ref", "unseen_obj"), default="unseen_ref")
    parser.add_argument("--reference-id", type=int, action="append")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/modulator_visualizations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-position", nargs=3, type=float, default=CAMERA_POSITION)
    parser.add_argument("--camera-target", nargs=3, type=float, default=CAMERA_TARGET)
    parser.add_argument("--fov", type=float, default=CAMERA_FOV)
    parser.add_argument("--_condition", choices=CONDITIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference_id is None:
        if args.dataset != "unseen_ref":
            parser.error("Choose test references with --reference-id for the unseen-object model")
        args.reference_id = list(DEFAULT_REFERENCES)
    if len(set(args.reference_id)) != len(args.reference_id):
        parser.error("Reference IDs must be unique")
    args.output_dir = args.output_dir.expanduser().resolve()
    os.chdir(ROOT)
    if args._condition:
        if len(args.reference_id) != 1:
            parser.error("The recording worker accepts exactly one reference")
        evaluate(args)
        return 0
    records = []
    for reference_id in args.reference_id:
        for condition in CONDITIONS:
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                "--dataset", args.dataset, "--reference-id", str(reference_id),
                "--output-dir", str(args.output_dir), "--device", args.device, "--seed", str(args.seed),
                "--camera-position", *map(str, args.camera_position),
                "--camera-target", *map(str, args.camera_target), "--fov", str(args.fov),
                "--_condition", condition]
            subprocess.run(command, cwd=str(ROOT), check=True)
        records.append(compose(args, reference_id))
    (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
