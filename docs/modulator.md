# Modulator training, inference and evaluation

A modulator is one PPO policy trained across a complete training partition.
It controls a frozen BC tracker; it is separate from the semantic specialists.
The unseen-reference and unseen-object experiments use 2,212 and 2,520 training
references respectively, with no test references included in training.

```bash
# Start with the provided BC tracker for each experiment
python scripts/train_modulator.py --dataset unseen_ref
python scripts/train_modulator.py --dataset unseen_obj

# Start with the BC tracker trained on the three example groups
python scripts/train_modulator.py --dataset examples \
  --tracker-checkpoint outputs/bc/examples/best.pth

# Resume model, PPO optimizer, normalization, epoch, LR and RNG
python scripts/train_modulator.py --dataset unseen_ref --resume
```

The default BC files are `checkpoints/bc_tracker_unseen_ref.pth` and
`checkpoints/bc_tracker_unseen_obj.pth`. Use `--tracker-checkpoint` for compatible
custom BC weights. Both must accept 273 observations and return 22 action means;
new BC checkpoints must match the selected training partition and latent hashes.
The BC parameters and observation normalization are frozen in evaluation mode.
PPO optimizes only the separate modulator model.

Defaults are one GPU (`--device cuda:0`), 8,192 environments, 30,000 PPO epochs and
seed 42. Training uses initial XY ±5 cm and yaw ±30 degree perturbations, domain
randomization, the existing observation noise and random initial frames. This
release explicitly uses these selected training perturbations. The old research
modulator task zeroed its initial-pose perturbation scale. Evaluation always
uses the reference initial pose, with initial XY/yaw perturbations disabled.

`configs/modulator.yaml` records the original Planner2k5 architecture and PPO
settings: separate actor/critic ELU MLPs of widths
4096, 2048, 1024, 512, 256, 128; learning rate `5e-4`, adaptive KL 0.016,
horizon 8, five mini-epochs, gamma 0.99 and tau 0.95.
Change `--num-envs`, `--epochs`, `--device`, `--seed` and `--output-dir` to set a
budget and output location. This training budget does not guarantee recreation
of historical checkpoint scores.

## Hierarchical control

Both policies receive 273-dimensional observations. The modulator returns 202
actions, clipped to [-1, 1], with the original ordering:

| Action indices (Python slices) | Meaning | Applied scale |
|---|---|---|
| `0:28` | Four future object xyz/xyzw references | Add `0.1 * 0.10 * action` |
| `28:116` | Four future 22-joint references | Add `0.1 * 0.314 * action` radians, then normalize by joint limits |
| `116:180` | Tracker's object representation | Replace the 64 latent values by `0.6 * action` |
| `180:202` | Tracker action residual | Add directly to the 22 BC means, then clip |

Quaternion reference offsets are additive, matching the research implementation.
The modulator sees the original observation. Reference and latent adjustments
enter the **next cached tracker observation**, while the residual acts in the
current step. `adadextrack/envs/modulator.py` implements this timing around the
existing 22-action physics task. `tracking_env.py` continues to compute the
original reward against the original reference; the reward is unchanged by the
choice of adjusted tracker targets.

## Pretrained checkpoints and inference

We provide `checkpoints/modulator_unseen_ref.pth` and
`checkpoints/modulator_unseen_obj.pth`. They are renamed byte-for-byte copies of
the original trained checkpoints. Use each with the matching
`bc_tracker_unseen_ref.pth` or `bc_tracker_unseen_obj.pth` and the published fixed
latents. The original paired trackers' model weights and normalization tensors
have been checked against the released BC trackers and are identical.

```bash
# Add the two pretrained modulators to an existing resource installation
python scripts/download_resources.py --components modulator_unseen_ref modulator_unseen_obj

# Test sets; defaults select the matching released modulator and BC
python scripts/infer_modulator.py --dataset unseen_ref --batch-size 553
python scripts/infer_modulator.py --dataset unseen_obj --batch-size 245

# Full-corpus visualization in reference batches
python scripts/infer_modulator.py --dataset unseen_ref \
  --ref-csv data/adadextrack_full/manifest.csv --batch-size 256 --render

# Training sets
python scripts/infer_modulator.py --dataset unseen_ref --split train --batch-size 256
python scripts/infer_modulator.py --dataset unseen_obj --split train --batch-size 256

# Evaluate a modulator and BC trained in the example pipeline
python scripts/infer_modulator.py --dataset examples \
  --checkpoint outputs/modulator/examples/best.pth \
  --tracker-checkpoint outputs/bc/examples/best.pth --render

# Evaluate a newly trained full-partition modulator
python scripts/infer_modulator.py --dataset unseen_ref \
  --checkpoint outputs/modulator/unseen_ref/best.pth --batch-size 553
```

The two full dataset modes default to the pretrained files in `checkpoints/`.
The example mode defaults to the user-trained files in `outputs/`. Use
`--checkpoint`, `--tracker-checkpoint` and `--latents-dir` for custom locations;
`--ref-csv` selects a custom reference manifest. Each modulator must use its
paired BC. The loader checks SHA256 and strictly loads the original tensors;
PPO optimizer/environment state is not restored for inference. Legacy files
are recognized using `checkpoints/checkpoints.json`, which also records the
network configuration, paired BC checksum and fixed latent hashes. Newly
trained models keep their metadata embedded in the checkpoint.

Evaluation shares the BC implementation: L2 object/wrist position errors,
quaternion angular errors, 16-joint MAE, Mean-error success and Completion over
all 300 frames. Initial XY/yaw perturbations, extra domain randomization and lag
are disabled, while observation noise remains. `--seed` defaults to 42.
Outputs include raw poses, `per_reference.csv` and `summary.json` under
`outputs/modulator_eval/<dataset>/<run>/`. Each batch uses a fresh subprocess.

## Training outputs and resume

Training writes `outputs/modulator/<dataset>/best.pth`, `last.pth`, the resolved
configuration, job metadata, status and PPO logs. These new checkpoints embed
architecture, training membership, paired BC SHA256, latent hashes, optimizer
and RNG state. The paired BC remains a separate file. Resume may increase the
total epoch budget but checks the remaining settings, tracker and representation.
