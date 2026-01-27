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

# --- Slurm srun resolution ---
SRUN="${SRUN:-/cm/shared/apps/slurm/current/bin/srun}"
if [ ! -x "$SRUN" ]; then
  echo "ERROR: srun not found/executable at SRUN='$SRUN'."
  echo "Set SRUN to the absolute path of srun and retry (e.g. SRUN=/path/to/srun sbatch ...)."
  exit 1
fi
export PATH="$(dirname "$SRUN"):$PATH"
echo "SRUN=$SRUN"
"$SRUN" --version || true
# --- End srun resolution ---

# --- Environment setup (edit before running) ---
# Replace `deepspeed` if your env name differs.
if command -v module >/dev/null 2>&1; then
  module purge
  module load cuda/12.9
  # Some module systems reset PATH; keep Slurm tools available.
  export PATH="$(dirname "$SRUN"):$PATH"
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

# --- Nsight Systems (NSYS=1) ---
mkdir -p "$RUN_DIR/profiles"
if [[ "${NSYS:-0}" == "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load cuda/12.9
  fi
  export NSYS_BIN=/gpfs/share/apps/cuda/12.9/bin/nsys
  if [ ! -x "$NSYS_BIN" ]; then
    echo "ERROR: NSYS=1 but NSYS_BIN is not executable: $NSYS_BIN"
    exit 1
  fi
  # Nsight Systems writes intermediate streams to /tmp by default; on some nodes /tmp can be constrained.
  export TMPDIR="${SLURM_TMPDIR:-$RUN_DIR/profiles/tmp}"
  mkdir -p "$TMPDIR"
  export NSYS_OUT_PREFIX="$RUN_DIR/profiles/nsys_${SLURM_JOB_ID}_%h"
  export NSYS_ARGS="profile --force-overwrite=true --sample=none --cpuctxsw=none --cuda-event-trace=false --trace=cuda,nvtx,osrt -o ${NSYS_OUT_PREFIX}"
  NSYS_WRAPPER="$RUN_DIR/profiles/nsys_wrapper.py"
  cat >"$NSYS_WRAPPER" <<'PY'
import os
import subprocess
import sys


def parse_local_rank(argv):
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        return int(env_local_rank)
    for arg in argv:
        if arg.startswith("--local_rank=") or arg.startswith("--local-rank="):
            return int(arg.split("=", 1)[1])
    return None


def main():
    local_rank = parse_local_rank(sys.argv[1:])
    if local_rank is None:
        local_rank = 0
    node_id = int(os.environ.get("SLURM_NODEID", "0") or "0")

    repo_root = os.environ.get("REPO_ROOT", os.getcwd())
    nsys_bin = os.environ.get("NSYS_BIN")
    nsys_args = os.environ.get("NSYS_ARGS", "")

    cmd = [sys.executable, "-u", os.path.join(repo_root, "src", "gpt2.py"), *sys.argv[1:]]

    # Profile only one rank on one node to keep overhead bounded and avoid end-of-job stalls.
    if local_rank == 0 and node_id == 0:
        if not nsys_bin or not os.path.exists(nsys_bin):
            raise SystemExit("NSYS=1 but NSYS_BIN is missing/unset in environment")
        nsys_cmd = [nsys_bin, *nsys_args.split(), *cmd]
        proc = subprocess.run(nsys_cmd, cwd=repo_root)
        raise SystemExit(proc.returncode)

    proc = subprocess.run(cmd, cwd=repo_root)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
PY
  export NSYS_WRAPPER
else
  export NSYS_ARGS=""
fi
# --- End NSYS setup ---

# --- Optional NCCL debug + topology artifacts (written under RUN_DIR) ---
# Enable once when needed:
#   NCCL_LOGS=1 sbatch scripts/slurm/run_2node_8gpu.sbatch
#
# Logs:
# - NCCL per-rank logs:   $RUN_DIR/nccl_rank_*.log
# - NCCL topology dump:   $RUN_DIR/nccl_topo.xml
#
# Quick grep tips:
# - Transport (IB vs socket):   grep -E "NET/|Using (IB|Socket)|bootstrap" nccl_rank_*.log
# - Rings/trees/channels:       grep -E "Channel|Rings|Trees|graph" nccl_rank_*.log
if [[ "${NCCL_LOGS:-0}" == "1" ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,NET,COLL
  export NCCL_DEBUG_FILE="$RUN_DIR/nccl_rank_%h_%p.log"
  export NCCL_TOPO_DUMP_FILE="$RUN_DIR/nccl_topo.xml"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_IB_HCA=mlx5_0
  export NCCL_SOCKET_IFNAME=^lo,docker0
fi
# --- End optional NCCL debug ---

HOSTS="$("$SRUN" --export=ALL --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" hostname | sort -u)"
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

# --- System evidence (once, on rank0 host) ---
if [ -n "${MASTER_ADDR:-}" ]; then
  "$SRUN" --export=ALL --nodes=1 --ntasks=1 --nodelist="$MASTER_ADDR" bash -lc '
    set -euo pipefail
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi topo -m > "$RUN_DIR/topo.txt"
    fi
    if command -v ibstat >/dev/null 2>&1; then
      ibstat > "$RUN_DIR/ibstat.txt"
    fi
  ' || true
fi
# --- End system evidence ---

# One Slurm task per node; torchrun spawns 4 local GPU processes per node (no SSH needed).
"$SRUN" --export=ALL bash -lc '
  set -euo pipefail
  source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
  conda activate deepspeed
  echo "[srun] which python: $(which python)"
  python -V

  TRAIN_SCRIPT="src/gpt2.py"
  if [[ "${NSYS:-0}" == "1" ]]; then
    TRAIN_SCRIPT="${NSYS_WRAPPER}"
  fi

  python -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=4 \
    --node_rank="$SLURM_PROCID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID:-gpt2_2node}" \
    "$TRAIN_SCRIPT" \
    --run_type optimized \
    --train_data_path train_small.bin \
    --val_data_path val_small.bin \
    --checkpoint_path "'"$RUN_DIR"'" \
    --epochs 1 \
    --seq_length 512 \
    --micro_batch_size_per_gpu 2 \
    --grad_accum_steps 1 \
    --deepspeed_config src/deepspeed_config.json
'

if [[ "${NSYS:-0}" == "1" ]]; then
  shopt -s nullglob
  for f in "$RUN_DIR"/profiles/nsys_${SLURM_JOB_ID}_*.qdrep "$RUN_DIR"/profiles/nsys_${SLURM_JOB_ID}_*.nsys-rep; do
    base="$(basename "$f")"
    host="$(echo "$base" | sed -E "s/^nsys_${SLURM_JOB_ID}_//; s/\\.(qdrep|nsys-rep)$//")"
    "$NSYS_BIN" stats --report gpu-kern-summary,gpu-api-summary,osrt-summary "$f" > "$RUN_DIR/profiles/nsys_stats_${host}.txt" || true
  done
fi
