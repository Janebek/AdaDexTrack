"""Recompute paper metrics from a BC run's saved 300-frame trajectories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from adadextrack.data.dataset import read_manifest
from adadextrack.data.reference import load_reference
from adadextrack.evaluation.metrics import compute_paper_metrics, aggregate_metrics, print_summary

ROOT=Path(__file__).resolve().parents[2]


def resolve_run_manifest(settings):
    """Resolve the exact recorded manifest after moving the project directory.

    Preserve split/group subdirectories; never substitute the full manifest for
    a recorded split or infer a dataset from reference ID ranges.
    """
    candidates=[]
    relative=settings.get("dataset_manifest_relative")
    if relative:
        relative=Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid recorded relative manifest path")
        candidates.append(ROOT/relative)
    recorded=Path(settings["reference_manifest"])
    parts=recorded.parts
    if "adadextrack_full" in parts:
        candidates.append(ROOT/"data"/Path(*parts[parts.index("adadextrack_full"):]))
    candidates.append(recorded)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("The inference run's exact dataset manifest is unavailable: "+str(recorded))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,help="Override the run's dataset manifest")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    source=json.loads((args.run_dir/"summary.json").read_text())
    if source["status"]!="complete":
        raise ValueError("Only complete inference runs can be evaluated")
    if args.manifest is None:
        args.manifest=resolve_run_manifest(source["settings"])
    expected_digest=source["settings"].get("dataset_manifest_sha256")
    if expected_digest and hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=expected_digest:
        raise ValueError("Dataset manifest differs from the inference run")
    entries={entry.reference_id:entry for entry in read_manifest(args.manifest,verify_hashes=True)}
    expected=source["settings"]["selected_reference_ids"]
    rows=[]
    for path in sorted((args.run_dir/"batches").glob("batch_*/poses_multi_ep00300.npz")):
        with np.load(str(path),allow_pickle=False) as data:
            if not np.array_equal(data["progress"],np.arange(1,301)) or not np.array_equal(data["reference_frames"],np.arange(300)):
                raise ValueError("Unexpected frame alignment: "+str(path))
            for index,ref_id in enumerate(data["reference_ids"].astype(str)):
                entry=entries[ref_id]
                if str(data["group_ids"][index])!=entry.group_id:
                    raise ValueError("Incorrect group metadata")
                metric=compute_paper_metrics(data["obj_poses"][:,index],data["hand_poses"][:,index],data["robot_qpos"][:,index],load_reference(entry.path))
                rows.append(dict(reference_id=ref_id,group_id=entry.group_id,object_name=entry.object_name,**metric))
    metrics=aggregate_metrics(rows,expected)
    order={ref_id:index for index,ref_id in enumerate(expected)}
    rows.sort(key=lambda row:order[row["reference_id"]])
    output=args.output_dir or args.run_dir/"reevaluation"
    output.mkdir(parents=True,exist_ok=False)
    with (output/"per_reference.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    grouped={}
    for field in ("group_id","object_name"):
        grouped[field]={}
        for value in sorted({row[field] for row in rows}):
            subset=[row for row in rows if row[field]==value]
            grouped[field][value]=dict(references=len(subset),**aggregate_metrics(subset,[row["reference_id"] for row in subset]))
    report=dict(status="complete",references=len(rows),metrics=metrics,by_group=grouped["group_id"],by_object=grouped["object_name"],source_run=str(args.run_dir))
    (output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for key,value in source["metrics"].items():
        if not np.isclose(value,metrics[key],rtol=1e-12,atol=1e-12):
            raise ValueError("Recomputed metric differs from online result: "+key)
    print_summary(metrics)
    print("[EVALUATION] Recomputed {} references; all metrics match the inference summary. {}".format(len(rows),output),flush=True)
    return 0
