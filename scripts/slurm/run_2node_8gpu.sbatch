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
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:-benchmarks/bigpurple_v100_2026-01-26/8gpu_2node}"
if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  echo "ERROR: cannot create RUN_DIR='$RUN_DIR' (permission denied)."
  echo "Set RUN_DIR to a writable path (e.g. \$SCRATCH/... or \$HOME/...) and resubmit:"
  echo "  RUN_DIR=\$SCRATCH/gpt2_runs/bigpurple_v100_2026-01-26/8gpu_2node sbatch scripts/slurm/run_2node_8gpu.sbatch"
  exit 1
fi

HOSTS="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$SLURM_NODELIST}")"
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
    --node_rank="$SLURM_NODEID" \
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
