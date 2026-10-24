# Specialist demonstrations and offline BC

[Back to the README](../README.md)

Train one PPO specialist per semantic group, collect its demonstrations, then
train **one BC tracker across all selected groups**. This module follows the
original collector and offline BC objective. It does not reject rollouts based
on tracking success. References are taken from the existing training manifests;
test references are excluded.

## Small-scale example

The default example uses groups 322 (stamp, 2 references), 369 (toothpaste,
4 references), and 435 (wineglass, 1 reference). The example preserves the same
seven members used by specialist training and replay.

```bash
# Train three independent semantic specialists from scratch
python scripts/train_specialists.py

# Collect demonstrations from their best.pth checkpoints
python scripts/collect_specialists.py

# Train one BC tracker from all three specialists' demonstrations
python scripts/train_bc.py

# Visualize and evaluate the trained BC tracker on the same seven references
python scripts/infer_bc.py \
  --ref-csv data/adadextrack_full/examples/specialists.csv \
  --checkpoint outputs/bc/examples/best.pth \
  --device cuda:0 --seed 42 --render
```

If the specialists have already been trained, start at collection. You can also
skip specialist training with the three released example checkpoints:

```bash
python scripts/download_resources.py --components example_specialists
python scripts/collect_specialists.py --specialist-root checkpoints/specialists
python scripts/train_bc.py
```

Use either your trained specialists or the released specialists for a collection
run. Choose a fresh `--output-dir` if the default demonstration directory already
exists. The released files contain exactly the seven example members;
they cannot replace specialists trained on full-split group membership.
See the [example checkpoint metrics](specialists.md#released-example-specialists).
The two published full-scale BC trackers remain available for independent inference.

## Full training partitions

| Data source | Semantic specialists | Training references | Output tracker |
|---|---:|---:|---|
| `unseen_ref` | 435 | 2,212 | `outputs/bc/unseen_ref/best.pth` |
| `unseen_obj` | 402 | 2,520 | `outputs/bc/unseen_obj/best.pth` |

Each full-split specialist uses **all references in its group within that training
partition**, rather than the selected example members. The two experiments have
separate specialists, demonstrations, and BC weights. They never combine their
training partitions.

```bash
# Train all 435 specialists on the unseen-reference training set
python scripts/train_specialists.py --dataset unseen_ref --all

# Collect demonstrations from all trained unseen-reference specialists
python scripts/collect_specialists.py --dataset unseen_ref --all

# Train one unseen-reference BC tracker
python scripts/train_bc.py --dataset unseen_ref

# Train all 402 specialists on the unseen-object training set
python scripts/train_specialists.py --dataset unseen_obj --all

# Collect demonstrations from all trained unseen-object specialists
python scripts/collect_specialists.py --dataset unseen_obj --all

# Train one unseen-object BC tracker
python scripts/train_bc.py --dataset unseen_obj
```

To evaluate either resulting tracker, pass its checkpoint and the matching
training or test CSV to `scripts/infer_bc.py`. For example:

```bash
python scripts/infer_bc.py \
  --ref-csv data/adadextrack_full/splits/unseen_trajectory_test.csv \
  --checkpoint outputs/bc/unseen_ref/best.pth --device cuda:0 --seed 42

python scripts/infer_bc.py \
  --ref-csv data/adadextrack_full/splits/unseen_object_test.csv \
  --checkpoint outputs/bc/unseen_obj/best.pth --device cuda:0 --seed 42
```

The example defaults are configurable training budgets. They are not a promise
to reproduce the published full-scale tracker checkpoints or their reported
accuracy.

## Collection settings and checkpoint discovery

By default, collection uses the first visible GPU, processes groups sequentially,
and loads `outputs/specialists/<dataset>/group_NNN/best.pth` automatically. Group
IDs are zero-padded to at least three digits. Before starting GPU workers, the
coordinator checks every selected checkpoint's dataset, group, reference IDs,
reference checksums, network, and normalization statistics. Missing or mismatched
specialists cause an error; they are never silently skipped.

```bash
# Train and collect using a different specialist output root
python scripts/train_specialists.py --dataset unseen_ref --all \
  --output-dir outputs/my_specialists
python scripts/collect_specialists.py --dataset unseen_ref --all \
  --specialist-root outputs/my_specialists

# Optional: independent groups on four GPUs, with automatic queue scheduling
python scripts/collect_specialists.py --dataset unseen_ref --all --gpus 0 1 2 3

# Change collection budget and exact output directory
python scripts/collect_specialists.py --envs-per-reference 10 --rounds 5 \
  --output-dir outputs/my_demonstrations
python scripts/train_bc.py --data-dir outputs/my_demonstrations \
  --output-dir outputs/my_bc
```

`--gpus` accepts any number of available GPU indices and follows the same
`CUDA_VISIBLE_DEVICES` indexing rules as specialist training. Each worker owns
one simulator and one specialist. A finished worker releases its GPU before the
next group starts. BC training uses one device selected by `--device`; its single
network is not split across the specialist GPU queue.

Collection retains specialist domain randomization and observation noise, with
initial object XY **±5 cm**, yaw **±30 degrees**, no action lag, and no random
start frame. The base seed is 42; a group's worker uses `seed + group_id`.
The teacher is frozen in evaluation mode, and its action mean is clipped to
`[-1, 1]` before execution. All rollouts are kept, including tracking failures.

The default budget is **20 environments per reference and 10 rounds**. All
environments within a group run together. The three example groups therefore use
40, 80, and 20 environments and produce 1,400 episodes in total.

An episode observes all 300 reference frames. The original environment returns
a startup observation before its first physical reset. Collection excludes that
action and the terminal observation's reset action. Valid training pairs are the
observations at progress 1 through 299 (reference frames 0 through 298) and their
teacher actions. This produces **299 pairs per episode**, or **418,600 pairs** for
the example defaults. No sample pairs an observation with an action from another
episode.

## Demonstration files

```text
outputs/bc_data/<dataset>/
  index.json
  group_NNN/
    job.json
    config.yaml
    worker.log
    status.json
    shards.json
    ref_<reference_id>_round_0000.npz
    ...
```

Each shard contains one reference for one round: float32 `obs` (N × 273), raw
teacher action means `actions` (N × 22), scalar reference/group/round IDs, and
per-sample reference frame, episode and environment IDs. Observations include
the environment's noise and clipping; they have not been normalized by the
teacher's running statistics. Action labels are clamped when computing BC loss.
Teacher sigmas are not needed by the original BC objective and are not saved.

The index records selected members, collection settings, teacher checkpoint
hashes, shard hashes, and sample counts. Only a complete collection can be used
for BC. Existing collection output directories are not overwritten; use a new
`--output-dir` for another collection run. Collection is not resumed partway
through an episode.

## Offline optimization and resume

BC runs with PyTorch and the bundled `rl_games`, without importing Isaac Gym or
creating a simulator. A randomly initialized student uses the published tracker
architecture: separate actor/critic ELU MLPs with widths **4096, 2048, 1024, 512,
256, 128**, with 273 observations and 22 actions. The student learns its own
observation running statistics and freezes them during inference. BC supervises
only the action mean; it adds no PPO, critic, or action-sigma objective.

The loss is the mean over samples of the **sum over 22 action dimensions** of
`(student_mean - clamp(teacher_mean, -1, 1))**2`. Defaults are Adam, learning rate
`5e-5`, zero weight decay, gradient norm limit 1, batch size 2,000, seed 42,
and 100 total epochs. Each epoch shuffles shards and their rows, visiting every
sample once. A final singleton batch is merged to keep normalization finite.

Shards are read on CPU with a bounded 256 MiB data cache, plus the active shard
and minibatches. Only the current minibatch is sent to the GPU. Storage scales
with reference count: default full-partition collection produces approximately
132 million (unseen-reference) or 151 million (unseen-object) observation/action
pairs. Adjust the collection budget to available disk space and training time.

```bash
# Adjust the BC training budget and CPU shard cache
python scripts/train_bc.py --dataset unseen_ref \
  --device cuda:0 --epochs 100 --batch-size 2000 \
  --learning-rate 5e-5 --cache-mb 256

# Continue to a total of 200 epochs using the same data and settings
python scripts/train_bc.py --dataset unseen_ref --epochs 200 --resume
```

Outputs are `outputs/bc/<dataset>/best.pth`, `last.pth`, `config.json`,
`metrics.jsonl`, and `status.json`. `best.pth` means lowest mean **training BC
loss**, not highest simulation success. Checkpoints include network metadata,
training reference IDs/checksums, model normalization, optimizer, epoch and RNG
state. Resume requires the same data and optimization settings; the total epoch
target may increase. New BC weights load through the same inference entry point
as the two released trackers.

Evaluation uses the shared [paper metrics](metrics.md): L2 position errors and
all 300 frames, with initial XY/yaw perturbations and additional domain
randomization disabled. Training collection keeps its perturbations enabled.
