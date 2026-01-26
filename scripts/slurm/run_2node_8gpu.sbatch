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
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
RUN_DIR="benchmarks/bigpurple_v100_${RUN_DATE}/8gpu_2node"
mkdir -p "$RUN_DIR"

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

# One process per GPU (4 per node) via Slurm; DeepSpeed initializes with env://.
srun --gpu-bind=single:1 bash -lc '
  set -euo pipefail
  export WORLD_SIZE="$SLURM_NTASKS"
  export RANK="$SLURM_PROCID"
  export LOCAL_RANK="$SLURM_LOCALID"

  python -u src/gpt2.py \
    --run_type optimized \
    --train_data_path train_small.bin \
    --val_data_path val_small.bin \
    --checkpoint_path "$RUN_DIR" \
    --epochs 1 \
    --seq_length 512 \
    --micro_batch_size_per_gpu 4 \
    --grad_accum_steps 1 \
    --deepspeed_config src/deepspeed_config.json \
    --local_rank "$SLURM_LOCALID"
'

