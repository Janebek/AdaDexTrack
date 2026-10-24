# Semantic specialist training

[Back to the README](../README.md)

One semantic group trains one independent PPO policy. Groups that share an
object still have separate policies and checkpoints. A group uses its selected
references across parallel environments; references never become separate
specialists.

## Example groups

The default example manifest is
`data/adadextrack_full/examples/specialists.csv`. It selects seven existing
unseen-trajectory training references, retaining their full-corpus IDs and
files:

| Group | Object | Reference IDs |
|---|---|---|
| 322 | stamp | 1237, 1232 |
| 369 | toothpaste | 1094, 1089, 1087, 1093 |
| 435 | wineglass | 2099 |

This is a small training example, not a separate benchmark or a new ID system.
Full-split training uses all members of each group within the chosen training
partition, rather than these selected example members.

## Replay and pretrained BC comparison

Run these commands from the repository root after installing the environment
and downloading resources:

```bash
# One group per window; all members of that group play together
python scripts/replay.py --group-id 322 --loops 3
python scripts/replay.py --group-id 369
python scripts/replay.py --group-id 435

# Play the three example groups in sequence
python scripts/replay.py --all

# Optional corpus overview: mix groups in 11 reference batches
python scripts/replay.py --dataset full --all --mode batch --batch-size 256 --overview

# Pretrained BC tracker on the same selected group members
python scripts/infer_bc.py \
  --ref-csv data/adadextrack_full/examples/specialists.csv \
  --group-id 322 --checkpoint checkpoints/bc_tracker_unseen_ref.pth \
  --device cuda:0 --seed 42 --render
```

Repeat the BC command with group 369 or 435 for the other two examples. Omit
`--render` on a server without a graphical desktop. Replay is kinematic playback
of references; policy inference runs the physics simulation and reports
tracking errors. Group mode remains the default for specialist comparisons.
The optional batch mode preserves manifest order, displays up to 256 references
per window, and does not change specialist training membership.

## Train from scratch

```bash
# Default: one visible GPU, three independent specialists trained sequentially
python scripts/train_specialists.py

# One example specialist
python scripts/train_specialists.py --group-id 322

# Full unseen-trajectory training split: 435 specialists, 2,212 references
python scripts/train_specialists.py --dataset unseen_ref --all

# Full unseen-object training split: 402 specialists, 2,520 references
python scripts/train_specialists.py --dataset unseen_obj --all
```

Each group starts with random network weights. Defaults are 8,192 environments,
seed 42, and **2,001 total PPO epochs per group**. An epoch collects eight control
steps per environment, then performs five PPO optimization passes. It is not a
single reference episode. Training stops at the epoch budget, saves the final
state, and moves to the next group; reward does not trigger early stopping.

The actor and critic have separate ELU MLPs with widths 1024, 512, 256, and 128.
The original specialist reward, PPO settings, observation noise and domain
randomization are retained. Initial object XY is sampled uniformly within
±5 cm per axis, and yaw within ±30 degrees. Training settings are in
`configs/specialist.yaml`, with shared simulation settings in
`configs/tracking.yaml`. Evaluation disables these initial perturbations and
additional domain randomization.

The environment and control are implemented in
[`tracking_env.py`](../adadextrack/envs/tracking_env.py). Its
`compute_hand_reward()` function defines the reward coefficients and episode
termination conditions.

```bash
# Change the training budget and output location
python scripts/train_specialists.py --group-id 322 \
  --num-envs 4096 --epochs 2001 --seed 42 --output-dir outputs/my_specialists

# Optional: one independent group per GPU, never distributed PPO for one group
python scripts/train_specialists.py --dataset unseen_ref --all --gpus 0 1

# Restrict visibility to one physical GPU
CUDA_VISIBLE_DEVICES=3 python scripts/train_specialists.py
```

`--gpus` indexes the current `CUDA_VISIBLE_DEVICES` list. Without `--gpus`, only
its first entry is used. When no visibility list is set, indices are physical
CUDA device indices. Reduce `--num-envs` if a specialist does not fit in GPU
memory. Each worker exits and releases its simulator before that GPU receives
another group. A failed group is recorded independently; the remaining groups
continue, and the command exits unsuccessfully if any group failed.

## Outputs and resume

The default output structure is:

```text
outputs/specialists/
  examples/
    group_322/
      best.pth
      last.pth
      job.json
      config.yaml
      status.json
      worker.log
      ppo/summaries/
    group_369/
    group_435/
  unseen_ref/group_322/
  unseen_obj/group_322/
```

`best.pth` follows the original best-reward selection after epoch 1500.
`last.pth` is saved every 500 epochs when episode statistics are available and
at completion. If the run ends before a best policy is selected, the final
policy also supplies `best.pth`. Training logs and TensorBoard summaries stay
inside each group's output directory.

```bash
# Resume incomplete groups; completed groups are skipped automatically
python scripts/train_specialists.py --resume

# Resume a full split with the same settings and output directory
python scripts/train_specialists.py --dataset unseen_ref --all --resume

# Resume one explicitly selected checkpoint into a new output directory
python scripts/train_specialists.py --group-id 322 \
  --resume-from outputs/specialists/examples/group_322/last.pth \
  --output-dir outputs/resumed_specialists
```

Checkpoints contain the model, optimizer, observation/value normalization,
epoch, frame count, learning rate, random-number states, network architecture,
group ID, and training reference IDs/checksums. Resume checks that the group,
reference membership, seed and training settings match. `--epochs` specifies
the total epoch target, including restored epochs; it can be increased when
resuming. The simulator starts fresh on resume; an in-progress physical
episode is not restored. A run interrupted before its first saved checkpoint
needs a new output directory to restart from scratch.

## Released example specialists

The optional resource component contains the three `best.pth` checkpoints,
their training configurations and per-reference evaluation metrics. Weights
retain their original training state and embedded network/group metadata.
Each run finished 2,001 PPO epochs; the selected checkpoint follows the
training-reward criterion described above.

| Group | Object | References | Selected epoch | T_o (cm) | R_o (rad) | T_h (cm) | R_h (rad) | E_finger (rad) | Mean-error (%) | Completion (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 322 | stamp | 2 | 1784 | 0.5154 | 0.191944 | 3.6169 | 0.224600 | 0.164760 | 100.00 | 100.00 |
| 369 | toothpaste | 4 | 1983 | 3.9818 | 0.328150 | 6.7369 | 0.291007 | 0.180596 | 100.00 | 79.67 |
| 435 | wineglass | 1 | 1998 | 0.4236 | 0.332047 | 2.1768 | 0.237023 | 0.150672 | 100.00 | 100.00 |

These are one-rollout-per-reference results on the example training members,
with seed 42, one environment per reference and the shared L2/full-300-frame
metrics. Initial XY/yaw perturbations, extra domain randomization and delay
are disabled; observation noise is retained. The toothpaste specialist has
incomplete framewise tracking despite passing the mean-error criterion.
These checkpoints provide a small collection/BC example.

```bash
python scripts/download_resources.py --components example_specialists

python scripts/infer_bc.py \
  --ref-csv data/adadextrack_full/examples/specialists.csv --group-id 322 \
  --checkpoint checkpoints/specialists/examples/group_322/best.pth \
  --device cuda:0 --seed 42 --render

# Collect from the released teachers, then train one example BC tracker
python scripts/collect_specialists.py --specialist-root checkpoints/specialists
python scripts/train_bc.py
```

Change both the group ID and checkpoint directory for groups 369 and 435.
Downloaded checkpoints are isolated from locally trained files under
`outputs/specialists/`. Their exact reference membership must match the
example manifest; use newly trained full-group specialists for the full splits.

## Infer with a trained specialist

The existing inference script reads the architecture embedded in specialist
checkpoints and still supports both published BC tracker checkpoints.

```bash
python scripts/infer_bc.py \
  --ref-csv data/adadextrack_full/examples/specialists.csv \
  --group-id 322 \
  --checkpoint outputs/specialists/examples/group_322/best.pth \
  --device cuda:0 --seed 42 --render
```

For groups 369 and 435, change both `--group-id` and the checkpoint directory.
Use the matching full training manifest for a specialist trained with
`--dataset unseen_ref` or `--dataset unseen_obj`. All policy comparisons use
the shared [L2 position and complete 300-frame metric definitions](metrics.md).
