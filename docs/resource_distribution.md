# Resource distribution

Code and website: https://github.com/Janebek/AdaDexTrack.
Runtime resource repository: https://huggingface.co/datasets/Janebek/AdaDexTrack.
The current upload status and pinned revision are in `configs/artifact_release.json`.

## Package layout

| File on Hugging Face | Restored location | Contents |
|---|---|---|
| `assets.tar.gz` | `assets/` | Object meshes, collision files, URDFs, xArm/LEAP model, texture and latents |
| `references.tar.gz` | `data/adadextrack_full/` | All 2,765 references, 444 group manifests, both train/test partitions and provenance |
| `bc_tracker_unseen_ref.pth` | `checkpoints/bc_tracker_unseen_ref.pth` | Unseen-trajectory BC tracker |
| `bc_tracker_unseen_obj.pth` | `checkpoints/bc_tracker_unseen_obj.pth` | Pretrained unseen-object BC tracker, 273 observations and 22 actions |
| `modulator_unseen_ref.pth` | `checkpoints/modulator_unseen_ref.pth` | Pretrained unseen-reference modulator, paired with `bc_tracker_unseen_ref.pth` |
| `modulator_unseen_obj.pth` | `checkpoints/modulator_unseen_obj.pth` | Pretrained unseen-object modulator, paired with `bc_tracker_unseen_obj.pth` |
| `object_ae.pth` | `checkpoints/object_ae.pth` | Optional original object AE, with encoder, decoder and training state |
| `object_ae_data.tar.gz` | `data/object_ae/` | Optional 57 input meshes, 55/2 train/validation split and SHA256 manifest |
| `example_specialists.tar.gz` | `checkpoints/specialists/examples/` | Optional pretrained stamp, toothpaste and wineglass specialists (groups 322, 369, 435), configs and metrics |
| `release.json` | Download metadata | Per-package and per-file SHA256 hashes, sizes and coverage |

Both unseen-reference (unseen-trajectory) and unseen-object BC trackers are included.
Two pretrained modulators are also included, paired with their matching BC trackers.
Three pretrained example specialists are available as an optional component.
The proprietary Isaac Gym SDK is not included.

## Download and restore

```bash
pip install 'huggingface_hub>=0.13.4'
python scripts/download_resources.py
python scripts/download_resources.py --components assets
```

The downloader defaults to the revision recorded in `configs/artifact_release.json`
(or `main` before the first upload). All files in a download are pinned to one
resolved Hub commit. A specific
version can be selected with `--revision <full commit hash>`. Files are verified
and unpacked into temporary staging before installation. Identical installed
files are preserved; conflicting local files cause a failure before any new file
is installed. Archives permit only regular files within each component's
declared runtime directory. An optional `--cache-dir` controls download caching.

For manual installation, extract the archives into the code repository root;
their entries already start with `assets/` and `data/adadextrack_full/`. Copy
both BC tracker files and both modulator files to `checkpoints/`. Check `release.json` before using manual copies.

## Optional example specialists

```bash
python scripts/download_resources.py --components example_specialists
python scripts/collect_specialists.py --specialist-root checkpoints/specialists
python scripts/train_bc.py
```

The three example specialists use groups 322 (stamp, two references), 369
(toothpaste, four references), and 435 (wineglass, one reference). Each was
trained for 2,001 PPO epochs. Their mean-error success rates are all 100%;
framewise Completion rates are 100%, 79.67% and 100%, respectively, using seed
42 and full 300-frame evaluation without initial XY/yaw perturbations.
Per-reference metrics and training settings accompany each checkpoint.
This component is opt-in and does not overwrite locally trained specialists.
See the code repository's `docs/specialists.md` for inference commands.

## Optional object autoencoder

The default download installs assets, references, both BC trackers and both modulators. AE weights
and AE input meshes are opt-in:

```bash
python scripts/download_resources.py --components object_ae object_ae_data
```

The original AE best checkpoint contains a 64-dimensional PointNet encoder and
2,048-point decoder, plus Adam/scheduler state. Its recorded budget is 1,000
epochs; it was selected after epoch 925. The 57 input meshes are split into
55 training inputs and two validation inputs: rubberduck and flute.
`data/object_ae/manifest.json` records the split and per-file hashes.
These inputs retain the source mesh terms described below.

The fixed 50 policy latents in `assets/latents` remain unchanged. Newly seeded
sampling/encoding does not promise to reproduce their bytes. See the code
repository's `docs/object_ae.md` for AE train, encode, decode and representation
compatibility. New representations require consistently retrained policies.

## Maintainer publication

```bash
python scripts/publish_huggingface.py --prepare-only
python scripts/publish_huggingface.py --verify-only
huggingface-cli login
python scripts/publish_huggingface.py
```

For large uploads, use a separate modern Python environment with the latest
`huggingface_hub` and its `hf_xet` dependency. The publisher does not need
Isaac Gym or PyTorch. This avoids changing the Python 3.8 simulation environment.
The current client uses `hf auth login`; older clients use `huggingface-cli login`.
Xet transfer can be accelerated with `HF_XET_HIGH_PERFORMANCE=1`.
If a slow proxy causes repeated response-timeout retries, increase the wait for
that upload process with `HF_XET_CLIENT_READ_TIMEOUT=1800s`; no global environment
or proxy change is required.

Packing is deterministic and includes only the explicit asset directory, full
reference dataset, both BC trackers, both modulators, original AE weights, AE inputs and the three example specialists. Logs, credentials, simulation outputs and
the research repository are not selected for upload. Packing checks the
checkpoints against their recorded SHA256 hashes and checks complete reference coverage.
Every archive member is read back and checked before upload. The official
`huggingface_hub` client uploads an explicit list of bundles, dataset card,
checksums and notices. Existing unrelated Hub files are not deleted; the upload
uses the observed parent commit to detect concurrent repository changes.

GitHub ignores runtime binaries. The Pages build copies only website inputs,
so these resources do not enter the website artifact.

## Source terms

Object meshes are included at the maintainer's request. Existing upstream
licenses and restrictions remain applicable. Inclusion does not establish that
additional redistribution permission has been obtained. No blanket BSD or
CC BY-NC license is assigned to the mixed bundle. See
[resource sources and terms](resource_licenses.md), also published as
`RESOURCE_LICENSES.md` on Hugging Face.

Generated references and trained weights are distinct from raw meshes. The
references are identified by the authors as DiffH2O-generated motions followed
by robot retargeting; this does not alone establish their licensing conditions.
The source-term review is not a statement that every component has received
separate redistribution permission.
