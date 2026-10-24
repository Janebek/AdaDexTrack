# Install AdaDexTrack on Ubuntu 20.04

[Back to the README](../README.md)

Follow the numbered steps in a Bash terminal. This guide starts with an Ubuntu
20.04 x86_64 machine and an NVIDIA GPU; no existing Conda environment is assumed.

## Software environment

| Component | Version / requirement |
|---|---|
| Tutorial target | Ubuntu 20.04, x86_64 |
| Development system | Ubuntu 20.04.6, NVIDIA driver 535.161.07 |
| Simulator | Standalone **Isaac Gym Preview 4** |
| Python | **3.8.18**, installed by Conda |
| PyTorch / torchvision | **2.4.1+cu121 / 0.19.1+cu121** |
| CUDA runtime | **12.1**, supplied by the PyTorch packages |
| NumPy | **1.23.5** |

This guide uses Ubuntu 20.04.6 for the released replay and BC tracker pipeline.
NVIDIA's archived Preview 4 supports Ubuntu 18.04/20.04 and Python 3.6–3.8. Use the Conda Python version specified here. See the
[official SDK archive](https://developer.nvidia.com/isaac-gym/download).

Use an NVIDIA GPU compatible with this pinned PyTorch/SDK stack. NVIDIA lists
Pascal or newer with at least 8 GB VRAM for Preview 4; this does not guarantee
that a 256-environment batch fits on every card. Start with an example semantic group;
reduce the BC inference `--batch-size` when necessary. Allow at least **20 GB free disk space**
for the environment, SDK, package caches and resources.

## 1. Install Ubuntu packages and the NVIDIA driver

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl ca-certificates \
  ubuntu-drivers-common libgl1 libglu1-mesa libglib2.0-0 \
  libx11-6 libxext6 libxcursor1 libxrandr2 libxinerama1 libxi6 \
  libvulkan1 vulkan-tools
```

Check the driver first:

```bash
nvidia-smi
```

If no working NVIDIA driver is installed, use Ubuntu's recommended driver and
reboot. These commands change the system driver; skip them on an already
configured machine with a suitable working driver.

```bash
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

After reboot, open a new terminal and run `nvidia-smi` again. CUDA 12.1 requires a
compatible Linux driver, with 525.60.13 as the minor-version compatibility
minimum; a supported **535 or newer** driver is a practical baseline for this
setup. Follow any Secure Boot key-enrollment prompts shown by Ubuntu.
Sources: [Ubuntu driver installation](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/),
[NVIDIA CUDA 12.1 compatibility](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html#cuda-driver).

The CUDA version displayed by `nvidia-smi` describes driver capability. The
Python environment's runtime version is checked separately in step 6. Running
this project's shipped C++ bridge does not require installing a separate
system CUDA Toolkit or `nvcc`; it needs the compiler from `build-essential`.

## 2. Install Conda

If Conda is already available, skip the installer and continue to step 3.
Otherwise install [Miniforge](https://conda-forge.org/download/):

```bash
mkdir -p "$HOME/Downloads"
cd "$HOME/Downloads"
curl -fL -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
conda --version
```

The `source` command activates Conda in the current terminal. `conda init bash`
enables it in subsequently opened Bash terminals. The install prefix above must
be unused; use your existing installation if that directory already exists.

## 3. Clone the project

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone https://github.com/Janebek/AdaDexTrack.git
cd AdaDexTrack
export ADADEXTRACK_ROOT="$PWD"
```

Keep this terminal open for the following steps. The default project directory
in this tutorial is `$HOME/projects/AdaDexTrack`.

## 4. Create the Python 3.8 environment

```bash
conda env create -f environment.yml
conda activate adadextrack
python --version
python -m pip --version
```

`python --version` should report **3.8.18**, and pip should belong to the same
Conda environment. `environment.yml` installs the pinned CUDA 12.1 PyTorch and
torchvision wheels from the official PyTorch index, NumPy, Ninja, Gym, SciPy,
imageio and the Hugging Face downloader. CUDA runtime dependencies are installed
with PyTorch. The [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v241)
document the CUDA 12.1 builds.

Install Isaac Gym in this project environment.

## 5. Install Isaac Gym Preview 4

Download the Isaac Gym Preview 4 release from the [NVIDIA website](https://developer.nvidia.com/isaac-gym).

```bash
cd isaacgym/python
pip install -e .
```

## 6. Set the runtime environment

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=2
```

## 7. Download resources

```bash
cd "$ADADEXTRACK_ROOT"
python scripts/download_resources.py
```

The public Hugging Face resources require no login. The downloader verifies
archive and per-file SHA256 checksums automatically and restores `assets/`,
`data/adadextrack_full/`, `checkpoints/bc_tracker_unseen_ref.pth` (unseen reference) and
`checkpoints/bc_tracker_unseen_obj.pth` (unseen object). No manual checksum
commands or individual reference downloads are needed.

## 8. Run the first replay and BC inference

On a machine with a working NVIDIA graphical desktop:

```bash
python scripts/replay.py --group-id 322 --loops 3
python scripts/infer_bc.py \
  --ref-csv data/adadextrack_full/examples/specialists.csv \
  --group-id 322 --device cuda:0 --render
```

For an SSH session or server without a display:

```bash
python scripts/infer_bc.py \
  --ref-csv data/adadextrack_full/examples/specialists.csv \
  --group-id 322 --device cuda:0
```

To process the full corpus:

```bash
# All references in 11 sequential windows
python scripts/replay.py --dataset full --all --mode batch --batch-size 256 --overview
python scripts/infer_bc.py --batch-size 256 --device cuda:0 --seed 42 --render
```

The viewer commands need a real graphical session. On headless machines, add
`--headless` to replay and omit `--render` from BC inference. Use a smaller
`--batch-size`, such as 64, for BC inference or replay in `--mode batch`
when GPU memory is insufficient. Replay defaults to `--mode group`, which
keeps each semantic group together; batch mode provides a corpus overview. Evaluation settings
remain the ones documented in the README; batching can change numerical results.

## 9. Train semantic specialists

```bash
# Three example groups, sequentially on the first visible GPU
python scripts/train_specialists.py
```

See the [specialist tutorial](specialists.md) for full-split training,
environment and epoch budgets, optional multiple GPUs, checkpoint resume and
inference with the trained policies.

## Starting again in a new terminal

```bash
conda activate adadextrack
cd "$HOME/projects/AdaDexTrack"
export ADADEXTRACK_ROOT="$PWD"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=2
```

The editable Isaac Gym installation persists. The two `export` commands are
terminal-local and should be repeated after environment activation.

## Troubleshooting

| Symptom | Check / fix |
|---|---|
| `conda: command not found` | Open a new terminal after `conda init bash`, or source `"$HOME/miniforge3/etc/profile.d/conda.sh"`. |
| `Unsupported Python version` / missing `gym_310` binding | Activate `adadextrack` and confirm Python 3.8.18; use that environment's `python -m pip`. |
| `libpython3.8.so.1.0` not found | Activate the environment and repeat the `LD_LIBRARY_PATH` export from step 6. |
| `np.float` / `np.int` missing | Check `python -m pip show numpy`; keep NumPy 1.23.5. |
| `Ninja is required` / `c++` not found | Confirm `ninja --version` and `c++ --version` in the activated terminal; install step 1 and recreate the pinned environment if dependencies were changed. |
| Compiler is killed while building `gymtorch` | Retry with `export MAX_JOBS=1` and sufficient RAM. |
| `torch.cuda.is_available()` is false | Check `nvidia-smi`, the active environment and the `+cu121` PyTorch build before starting a rollout. |
| Isaac Gym reports that PyTorch was imported first | Start a fresh Python process and import Isaac Gym before PyTorch. |
| Viewer fails / Vulkan uses an integrated GPU | Run `vulkaninfo --summary` from a graphical session and check the selected GPU. An Ubuntu-on-Xorg login session can help with legacy GLFW/viewer compatibility. |
| Multi-GPU / hybrid graphics select the wrong Vulkan device | If `/usr/share/vulkan/icd.d/nvidia_icd.json` exists, follow NVIDIA's SDK troubleshooting and set `VK_ICD_FILENAMES` to that file for this terminal. |
| Download stops | Re-run the download command; its cache can resume supported transfers. A checksum or existing-file conflict is reported explicitly. |

For other simulator/library errors, consult the downloaded SDK's `docs/index.html`
and `docs/install.html`. Keep the working system Python and driver libraries in
place while diagnosing Conda-specific import errors.
