# Big Purple / Slurm (Training Only)

This project supports Slurm runs on NYU Langone Big Purple. The scaling runner can use `$SLURM_TMPDIR` for fast local scratch and copies results back to your chosen output directory.

## sbatch Template (4 GPUs)

```bash
#!/bin/bash
#SBATCH --job-name=gpt2-scaling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module load cuda
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/1_download_data.py
python scripts/preprocess_small.py

python scripts/run_scaling_benchmarks.py \
  --epochs 1 \
  --seq_len 512 \
  --dataset_mode small \
  --repeat 3 \
  --out_dir /path/to/outputs/gpt2-scaling-${SLURM_JOB_ID} \
  --use_slurm_tmpdir
```

## Notes

- `$SLURM_TMPDIR` is used for checkpoint/data staging when `--use_slurm_tmpdir` is enabled (default). Results are copied back to `--out_dir`.
- Set `--repeat` to estimate variance; `scaling_report.json` includes mean and stdev.
- The training metrics JSON automatically captures Slurm metadata under the `scheduler` key.
