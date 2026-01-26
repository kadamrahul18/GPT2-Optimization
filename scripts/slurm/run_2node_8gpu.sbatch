#!/bin/bash
#
# 2 nodes x 4 GPUs/node (8 total) Slurm launcher.
# - Uses Slurm-native `srun` (no passwordless SSH required).
# - Writes artifacts to a single run dir via --checkpoint_path.
#
# Usage:
#   sbatch scripts/slurm/run_2node_8gpu.sbatch
#

#SBATCH --job-name=gpt2_2node_8gpu
#SBATCH --partition=gpu4_short
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

# --- Environment setup (edit before running) ---
# Replace `deepspeed` if your env name differs.
if command -v module >/dev/null 2>&1; then
  module purge
fi
source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate deepspeed
echo "which python: $(which python)"
python -V
python - <<'PY'
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda_device_count: {torch.cuda.device_count()}")
        print(f"cuda_device_name0: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"ERROR: torch/cuda check failed: {e}")
    raise
PY
# --- End environment setup ---

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/benchmarks/bigpurple_v100_$(date +%F)/8gpu_2node}"

echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unknown}"
echo "REPO_ROOT=$REPO_ROOT"
echo "PWD=$(pwd)"
echo "RUN_DIR=$RUN_DIR"
echo "REPO_ROOT perms:"
ls -ld "$REPO_ROOT" || true
echo "RUN_DIR parent perms:"
ls -ld "$(dirname "$RUN_DIR")" || true

if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  echo "ERROR: cannot create RUN_DIR='$RUN_DIR' (permission denied)."
  echo "Set RUN_DIR to a writable path (e.g. \$SCRATCH/... or \$HOME/...) and resubmit:"
  echo "  RUN_DIR=\$SCRATCH/gpt2_runs/bigpurple_v100_2026-01-26/8gpu_2node sbatch scripts/slurm/run_2node_8gpu.sbatch"
  exit 1
fi

HOSTS="$(srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" hostname | sort -u)"
if [ -z "${HOSTS//[[:space:]]/}" ]; then
  echo "ERROR: failed to discover hosts via srun; got empty host list."
  echo "SLURM_NNODES=${SLURM_NNODES:-unknown}"
  exit 1
fi
MASTER_ADDR="$(echo "$HOSTS" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29511}"

export MASTER_ADDR MASTER_PORT
export SLURM_HOSTS="$HOSTS"
export RUN_DIR
export GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"

echo "RUN_DIR=$RUN_DIR"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "HOSTS=${HOSTS//$'\n'/,}"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# One Slurm task per node; torchrun spawns 4 local GPU processes per node (no SSH needed).
srun bash -lc '
  set -euo pipefail
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=4 \
    --node_rank="$SLURM_PROCID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID:-gpt2_2node}" \
    src/gpt2.py \
    --run_type optimized \
    --train_data_path train_small.bin \
    --val_data_path val_small.bin \
    --checkpoint_path "$RUN_DIR" \
    --epochs 1 \
    --seq_length 512 \
    --micro_batch_size_per_gpu 2 \
    --grad_accum_steps 1 \
    --deepspeed_config src/deepspeed_config.json
'
