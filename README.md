# GPT-2 Training and Optimization with DeepSpeed

This project focuses on training and optimizing a GPT-2 model using PyTorch and DeepSpeed. It includes a baseline implementation using standard PyTorch training loops and an optimized version leveraging DeepSpeed's capabilities for pipeline parallelism, activation checkpointing, and other performance enhancements. The project also incorporates FlashAttention for improved attention mechanism efficiency.

## Overview

The core of this project is the implementation of a GPT-2 model, a powerful transformer-based language model. It includes training and validation routines, text generation capabilities, and comprehensive performance analysis. The project is designed to demonstrate the benefits of using DeepSpeed for large-scale model training, particularly in terms of memory management and computational efficiency.

## Main Features

- **GPT-2 Model Implementation:**
  - Custom GPT-2 model including token and positional embeddings, multi-head self-attention, feed-forward networks, and transformer blocks.
  - Support for FlashAttention for faster and more memory-efficient attention computation.
  - Language modeling head for text generation.

- **Training and Validation:**
  - Training loop with support for gradient accumulation, learning rate scheduling, and gradient clipping.
  - Validation loop to evaluate model performance on a separate dataset.
  - Checkpoint saving and loading for both baseline and optimized models.
  - Distributed training support using `torch.distributed`.

- **DeepSpeed Integration:**
  - Optimized training using DeepSpeed with configurable pipeline parallelism, activation checkpointing, and other optimizations.
  - Configuration via a JSON file for easy customization of DeepSpeed settings.

- **Performance Analysis:**
  - Profiling using `torch.profiler` to capture detailed performance metrics.
  - Kernel-level analysis to compare the performance of baseline and optimized models.
  - Metrics collection for training time, inference latency, inference throughput, maximum memory usage, and more.
  - Summary of results with a comparison between baseline and optimized runs.

- **Text Generation:**
  - Functionality to generate text based on a given prompt using the trained model.
  - Calculation of inference latency and throughput.

- **Data Preprocessing:**
  - `preprocess_data.py` script to download and preprocess the OpenWebText dataset into a binary format suitable for training.
  - Uses `tiktoken` for efficient tokenization and `datasets` for downloading and processing the dataset.

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- PyTorch 1.13.1 or later
- DeepSpeed
- Transformers library
- CUDA-enabled GPU (recommended)
- Other dependencies: `pytest`, `tqdm`, `numpy`, `tiktoken`, `datasets`

Data Preparation
The training and validation data are obtained from the OpenWebText dataset using the preprocess_data.py script.

This script downloads the dataset, tokenizes it using the GPT-2 tokenizer, and saves it into two binary files: train.bin (approximately 17GB) and val.bin (approximately 8.5MB).

Running the data preprocessing script:

python preprocess_data.py

This will create train.bin and val.bin in the project directory. These files will be used for training and validation.

Usage Examples
Baseline Training
To train the baseline GPT-2 model, use the following command:

python gpt2.py --run_type baseline --train_data_path train.bin --val_data_path val.bin --checkpoint_path checkpoint --epochs 1 --train_micro_batch_size_per_gpu 4 --gradient_accumulation_steps 4

Optimized Training with DeepSpeed
To train the optimized model with DeepSpeed, use the following command:

deepspeed gpt2.py --run_type optimized --train_data_path train.bin --val_data_path val.bin --checkpoint_path checkpoint --epochs 1 --deepspeed_config deepspeed_config.json --pipeline_stages 2 --train_micro_batch_size_per_gpu 4 --gradient_accumulation_steps 4

Note: Adjust the --pipeline_stages, --train_micro_batch_size_per_gpu, and --gradient_accumulation_steps parameters in the deepspeed_config.json file and the command-line arguments as needed for your specific hardware setup.

Text Generation
After training, you can generate text using the trained model. The script automatically generates text after training completion and prints it to the console. You can modify the prompt and max_length parameters in the main() function of gpt2.py to customize text generation.

Performance Analysis
The script automatically collects various performance metrics and saves them to JSON files in the checkpoint directory. After training, you can view a summary of the results, including a comparison between baseline and optimized runs, by examining the console output or the generated JSON files.
