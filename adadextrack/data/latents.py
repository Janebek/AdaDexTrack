"""Portable object-latent loading and provenance shared by policy entry points."""

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATENTS = ROOT / "assets/latents"


def latent_records(names, directory=DEFAULT_LATENTS):
    directory = Path(directory)
    paths = {}
    for path in directory.glob("*_latent.npy"):
        name = path.name[:-len("_latent.npy")].lower()
        if name in paths:
            raise ValueError("Ambiguous object latent: " + name)
        paths[name] = path
    result = {}
    for name in sorted(set(names)):
        path = paths.get(name.lower())
        if path is None:
            raise FileNotFoundError("Missing object latent: " + str(directory / (name+"_latent.npy")))
        value = np.load(str(path), allow_pickle=False)
        if value.shape != (64,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Expected finite float32[64] object latent: " + str(path))
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check_latent_metadata(metadata, names, directory=DEFAULT_LATENTS):
    actual = latent_records(names, directory)
    expected = (metadata or {}).get("latents", {})
    # Check the recorded training representation even when evaluation objects
    # are disjoint (the unseen-object experiment).
    recorded = latent_records(expected, directory) if expected else {}
    for name in expected:
        if recorded[name] != expected[name]:
            raise ValueError("Checkpoint used a different object latent: " + name)
    return actual
