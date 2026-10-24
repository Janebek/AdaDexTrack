#!/usr/bin/env python3
"""Evaluate a pretrained or user-trained modulator with full 300-frame metrics."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adadextrack import inference
from adadextrack.modulator_training import TRACKERS
from adadextrack.data.groups import ROOT, SOURCES

MODULATORS = {name: ROOT / ("checkpoints/modulator_" + name + ".pth")
              for name in ("unseen_ref", "unseen_obj")}


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("examples", "unseen_ref", "unseen_obj"), default="unseen_ref")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--tracker-checkpoint", type=inference.resolved_path,
                        help="Matching frozen BC tracker; defaults to the provided tracker for this dataset")
    args = inference.parse_args(argv, parser)
    args.policy_type = "modulator"
    def provided(option):
        return any(value == option or value.startswith(option + "=") for value in argv)
    if not provided("--checkpoint"):
        args.checkpoint = MODULATORS.get(args.dataset, ROOT / "outputs/modulator/examples/best.pth")
    if not provided("--ref-csv"):
        args.ref_csv = SOURCES[args.dataset]
        if args.dataset != "examples" and args.split == "test":
            args.ref_csv = args.ref_csv.with_name(args.ref_csv.name.replace("_train.csv", "_test.csv"))
    if not provided("--output-dir"):
        args.output_dir = ROOT / "outputs/modulator_eval" / args.dataset
    args.tracker_checkpoint = args.tracker_checkpoint or TRACKERS.get(args.dataset, ROOT / "outputs/bc/examples/best.pth")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    try:
        refs, versions = inference.preflight(args)
        inference.run_batches(args, refs, versions)
    except KeyboardInterrupt:
        print("[MOD-EVAL] Interrupted; completed batch outputs have been retained.", file=sys.stderr)
        return 130
    except Exception as error:
        print("[MOD-EVAL] ERROR: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
