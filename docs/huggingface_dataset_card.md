---
pretty_name: AdaDexTrack
license: other
license_name: component-specific-terms
license_link: https://huggingface.co/datasets/Janebek/AdaDexTrack/blob/main/RESOURCE_LICENSES.md
tags:
- robotics
- dexterous-manipulation
- motion-tracking
- 3d
language:
- en
size_categories:
- 1K<n<10K
---

# AdaDexTrack resources

Resources for **AdaDexTrack: Dynamic Modulation for Adaptive and Generalizable
Dexterous Manipulation Tracking**, CVPR 2026.

[Code and instructions](https://github.com/Janebek/AdaDexTrack) ·
[Project website](https://janebek.github.io/AdaDexTrack/) ·
[Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Adalibieke_AdaDexTrack_Dynamic_Modulation_for_Adaptive_and_Generalizable_Dexterous_Manipulation_Tracking_CVPR_2026_paper.pdf)

## Files

| File | Restored paths | Contents |
|---|---|---|
| `assets.tar.gz` | `assets/` | Object meshes and URDFs, xArm/LEAP robot model, floor texture, object latents |
| `references.tar.gz` | `data/adadextrack_full/` | All 2,765 reference motions, manifests, semantic groups, split lists and provenance |
| `bc_tracker_unseen_ref.pth` | `checkpoints/bc_tracker_unseen_ref.pth` | Pretrained unseen-trajectory BC tracker, 273 observations and 22 actions |
| `bc_tracker_unseen_obj.pth` | `checkpoints/bc_tracker_unseen_obj.pth` | Pretrained unseen-object BC tracker, 273 observations and 22 actions |
| `modulator_unseen_ref.pth` | `checkpoints/modulator_unseen_ref.pth` | Pretrained unseen-reference modulator, paired with `bc_tracker_unseen_ref.pth` |
| `modulator_unseen_obj.pth` | `checkpoints/modulator_unseen_obj.pth` | Pretrained unseen-object modulator, paired with `bc_tracker_unseen_obj.pth` |
| `object_ae.pth` | `checkpoints/object_ae.pth` | Optional original object AE, with encoder, decoder and training state |
| `object_ae_data.tar.gz` | `data/object_ae/` | Optional 57 input meshes, 55/2 train/validation split and SHA256 manifest |
| `example_specialists.tar.gz` | `checkpoints/specialists/examples/` | Optional pretrained stamp, toothpaste and wineglass specialists (groups 322, 369, 435), configs and metrics |
| `release.json` | Download metadata | Archive and per-file SHA256 checksums, sizes and coverage |

The corpus contains **2,765 motions, 50 objects, 444 semantic groups**, with 300
frames per motion. Reference IDs are `0–2764`; group IDs are `0–443`.
Each motion is stored once and shared by both experimental partitions.

| Partition | Train | Test |
|---|---:|---:|
| Unseen trajectory | 2,212 | 553 |
| Unseen object | 2,520 | 245 |

This package contains the full corpus, without height, checkpoint-success or
accuracy filtering. References were generated with DiffH2O and retargeted to the
robot. Their dictionaries contain object poses, robot joint states and link
transforms. Position units are meters, angles are radians and quaternions are xyzw.
The original NumPy dictionaries use pickle serialization; the project loader
reads these project resource files with `allow_pickle=True`.

Two BC trackers are released: `bc_tracker_unseen_ref.pth` is trained on the 2,212-reference
unseen-trajectory training split and is evaluated on `unseen_trajectory_test.csv`
(553 references); `bc_tracker_unseen_obj.pth` is trained on the 2,520-reference
unseen-object training split and is evaluated on `unseen_object_test.csv`
(245 references). Both have 273 observations and 22 actions. Each is intended as
the frozen tracker for its corresponding modulator experiment. Semantic
specialists for the three example groups are available as the optional
`example_specialists.tar.gz` component; full-split specialists are trained with
the code repository's specialist pipeline. The two
pretrained modulators are included as `modulator_unseen_ref.pth` and
`modulator_unseen_obj.pth`, each with 273 observations and 202 actions. Use each
with its matching released BC tracker and the fixed object latents.

## Download

From the code repository root, in the configured Python environment:

```bash
pip install 'huggingface_hub>=0.13.4'
python scripts/download_resources.py
```

The downloader resolves one Hub commit, verifies the archive and individual-file
checksums, and restores the paths above. It preserves identical installed files
and fails on conflicting local files. For a reproducible version, pass
`--revision <full Hub commit hash>`.

To download only the meshes, robot models, textures and object latents:

```bash
python scripts/download_resources.py --components assets
```

For manual installation, extract both `.tar.gz` archives into the **code
repository root** and put both `bc_tracker_unseen_ref.pth` and
`bc_tracker_unseen_obj.pth`, plus `modulator_unseen_ref.pth` and
`modulator_unseen_obj.pth`, in `checkpoints/`. Archives already
include their top-level `assets/` and `data/adadextrack_full/` paths.
Isaac Gym Preview 4 must be obtained and installed separately.

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

## Resource terms

There is **no single blanket license over this mixed resource bundle**.
The code repository's BSD license does not replace third-party asset terms.
See [RESOURCE_LICENSES.md](RESOURCE_LICENSES.md) for source information and
restrictions, and `notices/` for the retained available license files.

In particular, GRAB's published terms include noncommercial-use and
redistribution restrictions. Including files here does not grant additional
rights or establish permission from the original rightsholders. Generated
references and trained weights are distinct resources; no blanket relicensing
of third-party material is asserted.

## Evaluation

Run each tracker with its matching test split from the code repository root:

```bash
python scripts/infer_bc.py \
  --checkpoint checkpoints/bc_tracker_unseen_ref.pth \
  --ref-csv data/adadextrack_full/splits/unseen_trajectory_test.csv \
  --batch-size 553 --device cuda:0 --seed 42

python scripts/infer_bc.py \
  --checkpoint checkpoints/bc_tracker_unseen_obj.pth \
  --ref-csv data/adadextrack_full/splits/unseen_object_test.csv \
  --batch-size 245 --device cuda:0 --seed 42
```

Add `--render` for a viewer. `python scripts/download_resources.py` downloads
both BC trackers and both modulators by default; use `--components bc_tracker_unseen_ref bc_tracker_unseen_obj`
to download only the two weights.

Run the pretrained modulator with its matching frozen BC tracker:

```bash
python scripts/infer_modulator.py --dataset unseen_ref \
  --checkpoint checkpoints/modulator_unseen_ref.pth \
  --tracker-checkpoint checkpoints/bc_tracker_unseen_ref.pth --batch-size 553
python scripts/infer_modulator.py --dataset unseen_obj \
  --checkpoint checkpoints/modulator_unseen_obj.pth \
  --tracker-checkpoint checkpoints/bc_tracker_unseen_obj.pth --batch-size 245
```

Add `--render` for a viewer or `--split train` for the corresponding training set.

The code uses L2 position errors, quaternion angular errors and 16-finger-joint
MAE over all 300 frames. Initial object XY/yaw perturbations are disabled.
See the code repository for the exact Mean-error and Completion definitions.

```bibtex
@InProceedings{Adalibieke_2026_CVPR,
  author = {Adalibieke, Jianibieke and Han, Qianwei and Liu, Xueyi and Qin, Yuzhe and Yi, Li},
  title = {AdaDexTrack: Dynamic Modulation for Adaptive and Generalizable Dexterous Manipulation Tracking},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2026}
}
```
