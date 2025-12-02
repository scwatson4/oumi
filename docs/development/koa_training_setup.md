# Koa Oumi Training Environment Setup Guide

This guide captures the exact steps (and pitfalls) we encountered while preparing the Oumi training stack on UH Koa GPU nodes. It targets new researchers who need a working environment for running configs such as `configs/projects/dcvlr/starter_kit/scw71432-walton-hard-500-cosyn-traces-500.yaml`.

## 1. Prerequisites
- **Accounts**: Active Koa account with Duo MFA plus access to the UH VPN (if required off-campus).
- **SSH keys**: Upload your public key to `~/.ssh/authorized_keys` on Koa. Windows users should avoid `ControlMaster` tunneling; it caused repeated failures and had to be disabled.
- **GitHub access**: Either an SSH key added to GitHub or HTTPS credentials for cloning your fork of `oumi`.
- **Storage**: At least 5 GB of free space in `/mnt/lustre/koa/scratch/<username>` for pip caches and built wheels.

## 2. Login and module setup
```bash
ssh scwatson@koa.its.hawaii.edu
module load lang/Anaconda3/2024.02-1
source /opt/apps/software/lang/Anaconda3/2024.02-1/etc/profile.d/conda.sh
```
`conda init` is not permitted on shared nodes, so sourcing `conda.sh` is mandatory every new session.

## 3. Create the Python environment
```bash
conda create --name oumi python=3.12 -y
conda activate oumi
python -m pip install --upgrade pip
```
If `conda activate` complains about `Run 'conda init'`, you likely forgot to source `conda.sh` in step 2.

## 4. Install CUDA 12.1 PyTorch stack
FlashAttention requires a matching CUDA build. On Hopper-class GPUs with CUDA 12.1 toolkits available:
```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```
Verify with `python -c "import torch; print(torch.__version__)"` (expected `2.5.1+cu121` at the time of writing).

## 5. FlashAttention 2 installation
### 5.1 Required build helpers
```bash
python -m pip install psutil ninja packaging
```
Without these, the `flash-attn` build fails while generating metadata.

### 5.2 Avoiding cross-device wheel writes
Koa builds wheels in `/tmp` (local disk) but pip caches reside on Lustre, causing errors like `Invalid cross-device link`. Fix by keeping all build artifacts on the same filesystem:
```bash
mkdir -p /mnt/lustre/koa/scratch/$USER/{tmp,pip-cache}
export TMPDIR=/mnt/lustre/koa/scratch/$USER/tmp
export PIP_CACHE_DIR=/mnt/lustre/koa/scratch/$USER/pip-cache
```
Now install from source:
```bash
python -m pip install flash-attn --no-build-isolation
```
If the cross-device error persists (rare), download the prebuilt wheel directly to scratch and install it:
```bash
cd /mnt/lustre/koa/scratch/$USER/tmp
curl -LO https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/\
  flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
python -m pip install ./flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```
Validate with `python -m pip show flash-attn`.

## 6. Clone and install Oumi
```bash
cd /path/to/your/workdir
git clone git@github.com:<your-user>/oumi.git
cd oumi
python -m pip install -e .[train]
```
This installs the `oumi` CLI into the active conda environment. If `oumi` is "command not found", the editable install either failed or was run outside the environment.

## 7. Run training with your config
```bash
oumi train -c configs/projects/dcvlr/starter_kit/scw71432-walton-hard-500-cosyn-traces-500.yaml
```
Ensure the dataset paths referenced in the YAML exist on the node (e.g., mounted under `/mnt/lustre/koa/datasets/...`). Adjust `output_dir` to a scratch location you own.

## 8. Known issues and resolutions
| Symptom | Cause | Fix |
| --- | --- | --- |
| `conda activate` demands `conda init` | Shared module disallows `conda init` | Source `.../conda.sh` each session. |
| `ModuleNotFoundError: torch` while installing `flash-attn` | Torch not yet installed | Install CUDA-matched Torch before FlashAttention. |
| `ModuleNotFoundError: psutil` during FlashAttention metadata build | Missing build helper | `python -m pip install psutil ninja packaging`. |
| `error: [Errno 18] Invalid cross-device link` when building wheel | Pip tmp/cache on different filesystems | Set `TMPDIR` and `PIP_CACHE_DIR` to the same scratch location or install from a downloaded wheel. |
| `oumi: command not found` | CLI not installed into env | Run `python -m pip install -e .[train]` inside the `oumi` repo with the env active. |

## 9. Verification checklist
- `which python` shows `~/.conda/envs/oumi/bin/python`.
- `python -m pip show torch` reports a CUDA build (e.g., `2.5.1+cu121`).
- `python -m pip show flash-attn` returns `2.8.3`.
- `oumi --help` prints CLI usage.
- `nvidia-smi` shows the expected GPUs when running inside the job allocation.

Following the sequence above reproduced a working environment end-to-end. Deviating (e.g., running `pip install` before activating the env, or letting pip use `/tmp` on a different filesystem) caused the failures captured in the table.
