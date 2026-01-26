# GPT-2 Training Benchmarks (Training Only)

Train GPT-2 with a baseline single process or multi-GPU DeepSpeed on a single node. Each run writes `training_metrics.json` (schema v2.0) and `RUN_COMPLETE.txt`.

## Key Results (Big Purple V100, Fixed-Work)

Artifacts:
- `benchmarks/bigpurple_v100_2026-01-08/1gpu/training_metrics.json`
- `benchmarks/bigpurple_v100_2026-01-08/2gpu/training_metrics.json`
- `benchmarks/bigpurple_v100_2026-01-08/4gpu/training_metrics.json`

Throughput is **global tokens/sec**. Wall time is the fixed-work total time.

| GPUs | Tokens/sec (global) | Wall time (s) | Speedup vs 1 GPU | Scaling efficiency |
| --- | --- | --- | --- | --- |
| 1 | 29600.05 | 1017.61 | 1.00 | 1.00 |
| 2 | 57289.31 | 522.01 | 1.94 | 0.97 |
| 4 | 100791.44 | 296.10 | 3.41 | 0.85 |

Regenerate table:
```bash
python scripts/generate_scaling_table.py \
  --gpu1 benchmarks/bigpurple_v100_2026-01-08/1gpu/training_metrics.json \
  --gpu2 benchmarks/bigpurple_v100_2026-01-08/2gpu/training_metrics.json \
  --gpu4 benchmarks/bigpurple_v100_2026-01-08/4gpu/training_metrics.json
```

## Benchmark Protocol

Constants (fixed work):
- `seq_len = 512`
- `global_batch_size = 16`
- `optimizer_steps = 3561`
- `tokens_processed_global = 29,171,712`
- `micro_batch_size_per_gpu = 4` with `grad_accum_steps` set per GPU count

Global batch size is kept constant by reducing `grad_accum_steps` as GPU count increases.

## Quickstart

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data:
```bash
python scripts/1_download_data.py
python scripts/preprocess_small.py
```

Runs (constant global batch size = 16):
```bash
# 1 GPU baseline
export CUDA_VISIBLE_DEVICES=0
python src/gpt2.py \
  --run_type baseline \
  --train_data_path train_small.bin \
  --val_data_path val_small.bin \
  --checkpoint_path checkpoint/1gpu \
  --epochs 1 \
  --seq_length 512 \
  --micro_batch_size_per_gpu 4 \
  --grad_accum_steps 4 \
  --deepspeed_config src/deepspeed_config.json

# 2 GPU DeepSpeed
export CUDA_VISIBLE_DEVICES=0,1
deepspeed src/gpt2.py \
  --run_type optimized \
  --train_data_path train_small.bin \
  --val_data_path val_small.bin \
  --checkpoint_path checkpoint/2gpu \
  --epochs 1 \
  --seq_length 512 \
  --micro_batch_size_per_gpu 4 \
  --grad_accum_steps 2 \
  --deepspeed_config src/deepspeed_config.json

# 4 GPU DeepSpeed
export CUDA_VISIBLE_DEVICES=0,1,2,3
deepspeed src/gpt2.py \
  --run_type optimized \
  --train_data_path train_small.bin \
  --val_data_path val_small.bin \
  --checkpoint_path checkpoint/4gpu \
  --epochs 1 \
  --seq_length 512 \
  --micro_batch_size_per_gpu 4 \
  --grad_accum_steps 1 \
  --deepspeed_config src/deepspeed_config.json
```

## 2-node quickstart

Slurm (2 nodes × 4 GPUs/node, 8 total):
```bash
sbatch scripts/slurm/run_2node_8gpu.sbatch
```

By default, outputs land in `benchmarks/bigpurple_v100_2026-01-26/8gpu_2node/`. Override with `RUN_DIR=...` when submitting if desired.

## Artifacts

- `training_metrics.json`: per-epoch tokens/sec, epoch wall time, batch config, and completion metadata.
- `RUN_COMPLETE.txt`: completion marker with timestamp.

## Notes / Limitations

- Charts are optional and use only `training_metrics.json` (`scripts/3_generate_charts.py`).
- Slurm metadata is captured in `training_metrics.json` when present.
