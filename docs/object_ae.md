# Object autoencoder and latents

The point autoencoder has a PointNet encoder (3 → 64 → 128 → 256 → 512 → 1024,
max pooling, 1024 → 64) and an MLP decoder (64 → 512 → 1024 → 6144, tanh).
The decoder returns 2,048 points. Parameter names and shapes match the original
checkpoint. This module uses PyTorch and Trimesh independently of Isaac Gym.

## Install the optional resources

From the repository root, in the configured environment:

```bash
pip install -e '.[object-ae]'
python scripts/download_resources.py --components object_ae object_ae_data
```

`checkpoints/object_ae.pth` is the original `best.pth`, including encoder,
decoder, Adam and scheduler state. Its SHA256 is
`71e6057345b3d230363f41819cda25fcc3bbb66adb12d12fdcb30badfab33b60`.
The stored `epoch=926` denotes the next epoch: 925 epochs had completed when
this best checkpoint was saved. The original run budget was 1,000 epochs.

`data/object_ae/manifest.json` lists **57 input meshes**, their sizes,
SHA256 hashes, and the training/validation split. `SHA256SUMS` contains the same
file hashes. The split contains 55 training inputs and two validation inputs:
**rubberduck and flute**. The 57 AE inputs are distinct from the policy corpus's
50 objects and from its trajectory/object train/test partitions.

## Encode and decode

```bash
python scripts/object_ae.py encode \
  --checkpoint checkpoints/object_ae.pth \
  --input data/object_ae/meshes \
  --output-dir outputs/object_ae/latents --seed 42

python scripts/object_ae.py decode \
  --checkpoint checkpoints/object_ae.pth \
  --latent-dir assets/latents \
  --output-dir outputs/object_ae/reconstruction
```

Encoding samples 2,048 surface points with an explicit Trimesh RNG seed, subtracts
the sampled-point mean and divides by the maximum radius. Vertex-only inputs
are resampled with an explicit NumPy generator. The encoder maps this normalized
point cloud to a 64-dimensional float32 vector, saved as `<object>_latent.npy`.
`encoding.json` records input/output hashes, sampling seed, normalization center
and scale, library versions, device and checkpoint checksum. Existing latent files
are never overwritten. Decoding exports PLY **point clouds in normalized
coordinates**; it does not restore an object's metric size or position.

The published `assets/latents` files remain the fixed representations used for
paper evaluation. They match the original policy resources byte for byte.
The original surface sampler did not receive an explicit seed; new deterministic
sampling and encoding do not promise to reproduce those old latent bytes.
Use the published latents together with the published BC checkpoints.

## Train and resume

```bash
python scripts/object_ae.py train

# Resume an interrupted new run, keeping its original total epoch budget
python scripts/object_ae.py train --resume outputs/object_ae/train/last.pth

# Continue from the original released best checkpoint in a new output directory
python scripts/object_ae.py train --resume checkpoints/object_ae.pth \
  --output-dir outputs/object_ae/continued
```

Defaults follow the original checkpoint configuration: 1,000 epochs, batch size
8 with incomplete training batches dropped, 2,048 points, latent dimension 64,
Adam with learning rate `1e-3` and zero weight decay, CosineAnnealingLR over the
full epoch budget, seed 42, validation fraction 0.05 and four data workers.
The loss is the sum of the two directional mean nearest-neighbor **squared L2**
distances. `--cdist-chunk N` computes this same loss in smaller chunks.

Training samples use a seed derived from the global seed, epoch and sorted input
index, independent of worker scheduling. Checkpoints contain model, optimizer,
scheduler, next epoch, best validation loss, input/split provenance and RNG state.
Resume checks configuration and provenance. Original checkpoints lack the newer
provenance/RNG records, so they restore recorded training state while starting the
new explicit sampling convention. New runs are reproducible within the same
software/device configuration; they do not claim bitwise reproduction of the
historical stochastic run. Outputs are `best.pth`, `last.pth` and `provenance.json`.
Use `--device cpu` for CPU execution and CLI options to change the training budget.

## Use a newly trained representation

```bash
python scripts/object_ae.py encode \
  --checkpoint outputs/object_ae/train/best.pth \
  --input data/object_ae/meshes --output-dir outputs/object_ae/new_latents --seed 42

python scripts/train_specialists.py --dataset examples \
  --latents-dir outputs/object_ae/new_latents --output-dir outputs/new_repr/specialists
python scripts/collect_specialists.py --dataset examples \
  --specialist-root outputs/new_repr/specialists \
  --latents-dir outputs/object_ae/new_latents --output-dir outputs/new_repr/bc_data
python scripts/train_bc.py --dataset examples \
  --data-dir outputs/new_repr/bc_data --latents-dir outputs/object_ae/new_latents \
  --output-dir outputs/new_repr/bc
python scripts/train_modulator.py --dataset examples \
  --tracker-checkpoint outputs/new_repr/bc/best.pth \
  --latents-dir outputs/object_ae/new_latents --output-dir outputs/new_repr/modulator
```

All stages must use the same representation. Offline BC learns the latents already
embedded in collected observations; its optional `--latents-dir` checks their
provenance rather than replacing observation columns. Specialist, collection,
BC and modulator checkpoints record latent hashes. Inference accepts the same
`--latents-dir` and rejects known mismatches. A different AE requires a newly
trained specialist → BC → modulator stack; the provided policies should continue
to use the provided latents.
