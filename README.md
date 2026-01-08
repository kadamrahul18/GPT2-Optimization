# Accelerating GPT-2 Training on Multi-GPU Systems with DeepSpeed

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project demonstrates a successful strategy for drastically reducing the training time of a GPT-2 model by scaling from a single GPU to a multi-GPU cluster on **Amazon Web Services (AWS)** using the **DeepSpeed** library.

The primary outcome was a **71% reduction in training time (a 3.5x speedup)**, showcasing a robust and reproducible MLOps workflow for distributed training.

## The Core Achievement: A 71% Reduction in Training Time

By implementing a Data Parallelism strategy with DeepSpeed, the time required to train the model for one epoch was reduced from **~38 minutes to under 11 minutes**.

![Training Time Comparison](docs/training_time_comparison.png)

| Configuration | Training Time | Speedup |
| :--- | :--- | :--- |
| Baseline (1x T4) | ~2259 seconds | 1x |
| **Optimized (4x T4s)** | **~650 seconds** | **~3.5x** |

## The Engineering Challenge

The primary bottleneck in many deep learning projects is the long training time required on a single GPU. This slow feedback loop hinders experimentation, hyperparameter tuning, and model development. The objective of this project was to directly address this challenge by implementing + benchmarking a scalable, distributed training solution.

## Technical Solution & Architecture

The performance gains were achieved through a combination of key technologies:

1.  **PyTorch:** The foundational deep learning framework used for the GPT-2 model implementation.
2.  **DeepSpeed:** A powerful library from Microsoft used to implement a **Data Parallelism** strategy. DeepSpeed handled the complexities of distributing data batches, synchronizing gradients, and managing optimizer states across all four T4 GPUs.
3.  **Amazon Web Services (AWS) EC2:** The experiments were conducted on a `g4dn.12xlarge` instance (4x NVIDIA T4 GPUs) to simulate a common cloud-based production environment.

### An Important Engineering Decision
The initial codebase included a custom Triton-based Flash Attention kernel intended for newer GPU architectures. During testing on the T4 hardware, this kernel failed due to a hardware-software incompatibility. Instead of getting blocked, I made the pragmatic engineering decision to **disable the non-essential custom kernel** and focus on the primary optimization: the multi-GPU distributed framework. This ensured system stability and still delivered massive performance improvements, demonstrating an ability to adapt solutions for real-world hardware constraints.

## DeepSpeed Config Notes

- Pipeline parallelism is disabled in `src/deepspeed_config.json` because the training code does not use a DeepSpeed `PipelineModule`, so enabling it would be misleading.
- `wall_clock_breakdown` is enabled to expose DeepSpeed timing breakdowns for training performance transparency.

## Training Benchmark Protocol

- 1 GPU baseline uses the standard Python entrypoint; 2 and 4 GPU runs use DeepSpeed.
- Fixed `seq_len`, fixed dataset subset size, and fixed hyperparameters across runs.
- On Slurm, results include scheduler metadata in `training_metrics.json`.
- For constant global batch size = 16:
  - 1 GPU: `--micro_batch_size_per_gpu 4 --grad_accum_steps 4`
  - 2 GPU: `--micro_batch_size_per_gpu 4 --grad_accum_steps 2`
  - 4 GPU: `--micro_batch_size_per_gpu 4 --grad_accum_steps 1`
- CLI batch flags override values in `src/deepspeed_config.json`.

## Reproduce Results

The results of this project are fully reproducible. The training scripts automatically generate verifiable `training_metrics.json` files, and a separate script consumes these files to generate the chart shown above.

### 1. Setup
```bash
git clone https://github.com/kadamrahul18/GPT2-Optimization.git
cd GPT2-Optimization
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
python scripts/1_download_data.py
python scripts/preprocess_small.py
```

### 3. Run the Single-GPU Baseline (Local)
This run will produce `checkpoint/scaling_runs/1gpu_baseline/<run_id>/training_metrics.json` when using the scaling script, or `checkpoint/baseline_t4_small/training_metrics.json` if run manually.
```bash
export CUDA_VISIBLE_DEVICES=0
time python src/gpt2.py \
    --run_type baseline \
    --train_data_path train_small.bin \
    --val_data_path val_small.bin \
    --checkpoint_path checkpoint/baseline_t4_small \
    --epochs 1 \
    --seq_length 512 \
    --deepspeed_config src/deepspeed_config.json
```

### 4. Run the Multi-GPU Optimized Version (Local, 2 or 4 GPUs)
This run will produce `checkpoint/optimized_t4_small/training_metrics.json` if run manually.
```bash
export CUDA_VISIBLE_DEVICES=0,1
time deepspeed src/gpt2.py \
    --run_type optimized \
    --train_data_path train_small.bin \
    --val_data_path val_small.bin \
    --checkpoint_path checkpoint/optimized_t4_small \
    --epochs 1 \
    --seq_length 512 \
    --deepspeed_config src/deepspeed_config.json
```

### 5. Run the Scaling Benchmarks (Local, 1/2/4 GPUs)
This command runs all three configurations on a single machine and produces `scaling_report.json`.
```bash
python scripts/run_scaling_benchmarks.py \
    --epochs 1 \
    --seq_len 512 \
    --dataset_mode small \
    --deepspeed_config src/deepspeed_config.json
```

### 6. Run the Scaling Benchmarks (Slurm)
See `docs/HPC_SLURM.md` for an sbatch template and usage on Big Purple.

### 7. Generate the Charts
This command consumes the output of the manual baseline + optimized runs to create training-only visuals.
```bash
pip install matplotlib seaborn
python scripts/3_generate_charts.py \
    --baseline-json checkpoint/baseline_t4_small/training_metrics.json \
    --optimized-json checkpoint/optimized_t4_small/training_metrics.json
```

## Reproducible Scaling Benchmarks (Training)

Artifacts used for the table below:
- `benchmarks/bigpurple_v100_2026-01-08/1gpu/training_metrics.json`
- `benchmarks/bigpurple_v100_2026-01-08/2gpu/training_metrics.json`
- `benchmarks/bigpurple_v100_2026-01-08/4gpu/training_metrics.json`

To regenerate the table:
```bash
python scripts/generate_scaling_table.py \
  --gpu1 benchmarks/bigpurple_v100_2026-01-08/1gpu/training_metrics.json \
  --gpu2 benchmarks/bigpurple_v100_2026-01-08/2gpu/training_metrics.json \
  --gpu4 benchmarks/bigpurple_v100_2026-01-08/4gpu/training_metrics.json
```

Constants (fixed-work protocol):
- `seq_len = 512`
- `global_batch_size = 16`
- `optimizer_steps = 3561`
- `tokens_processed_global = 29,171,712`
- `micro_batch_size_per_gpu = 4` with `grad_accum_steps` set per GPU count (below)

Throughput is **global tokens/sec**, and wall time is the fixed-work total time for the run.

| GPUs | Tokens/sec (global) | Wall time (s) | Speedup vs 1 GPU | Scaling efficiency |
| --- | --- | --- | --- | --- |
| 1 | 29600.05 | 1017.61 | 1.00 | 1.00 |
| 2 | 57289.31 | 522.01 | 1.94 | 0.97 |
| 4 | 100791.44 | 296.10 | 3.41 | 0.85 |

Git commits in these artifacts:
- 1 GPU: `efa044856c95a3b8bc8791370452d65df9767723`
- 2/4 GPU: `bfcef35ad77bf757c0b05c0f04626233572744d5`

How to reproduce (constant global batch size = 16 by adjusting grad accumulation):
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

Global batch size is kept constant by reducing `grad_accum_steps` as GPU count increases.

## Artifacts

- `training_metrics.json` includes a versioned training-only schema with per-epoch `tokens_per_sec_global`, `epoch_wall_time_sec`, rank-0 CUDA memory stats, and completion metadata.
- `RUN_COMPLETE.txt` is written when a run finishes so automation can detect completion.
- `scaling_report.json` aggregates per-run throughput and epoch time. It computes:
  - `speedup_vs_1gpu = throughput_N / throughput_1gpu` (or inverse epoch time if throughput is missing)
  - `scaling_efficiency = speedup_vs_1gpu / N`
---
