import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone


def safe_get_git_commit(repo_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def sha256_file(path):
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return "unknown"


def get_python_version():
    return platform.python_version()


def get_host_info():
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
    }


def get_library_versions():
    versions = {}
    try:
        import torch  # noqa: F401
        versions["torch"] = torch.__version__
        versions["cuda"] = torch.version.cuda or "unknown"
    except Exception:
        versions["torch"] = "unknown"
        versions["cuda"] = "unknown"

    try:
        import deepspeed  # noqa: F401
        versions["deepspeed"] = deepspeed.__version__
    except Exception:
        versions["deepspeed"] = "unknown"

    try:
        import transformers  # noqa: F401
        versions["transformers"] = transformers.__version__
    except Exception:
        versions["transformers"] = "unknown"

    return versions


def get_hardware_info(world_size):
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
        else:
            gpu_name = "none"
            gpu_count = 0
    except Exception:
        gpu_name = "unknown"
        gpu_count = 0

    return {
        "gpu_name": gpu_name,
        "gpu_count_visible": gpu_count,
        "world_size": world_size,
    }


def selected_deepspeed_fields(ds_config):
    optimizer = ds_config.get("optimizer", {})
    optimizer_params = optimizer.get("params", {})
    zero_opt = ds_config.get("zero_optimization", {})
    fp16 = ds_config.get("fp16", {})
    scheduler = ds_config.get("scheduler", {})
    scheduler_params = scheduler.get("params", {})

    return {
        "train_micro_batch_size_per_gpu": ds_config.get("train_micro_batch_size_per_gpu"),
        "gradient_accumulation_steps": ds_config.get("gradient_accumulation_steps"),
        "zero_stage": zero_opt.get("stage"),
        "fp16_enabled": fp16.get("enabled"),
        "optimizer_type": optimizer.get("type"),
        "optimizer_lr": optimizer_params.get("lr"),
        "scheduler_type": scheduler.get("type"),
        "scheduler_warmup_num_steps": scheduler_params.get("warmup_num_steps"),
    }


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def build_initial_metrics(args, ds_config, train_dataset, val_dataset, world_size, repo_root, ds_config_path):
    host_info = get_host_info()
    library_versions = get_library_versions()
    hardware = get_hardware_info(world_size)
    hardware["cuda_version"] = library_versions.get("cuda", "unknown")
    hardware["instance_type"] = "g4dn.12xlarge"

    train_sequences = len(train_dataset) if train_dataset is not None else 0
    val_sequences = len(val_dataset) if val_dataset is not None else 0

    seq_length = args.seq_length
    micro_batch = ds_config.get("train_micro_batch_size_per_gpu", args.train_micro_batch_size_per_gpu)
    grad_accum = ds_config.get("gradient_accumulation_steps", 1)
    global_batch = micro_batch * grad_accum * world_size
    optimizer = ds_config.get("optimizer", {})
    optimizer_lr = optimizer.get("params", {}).get("lr")
    precision = "fp16" if ds_config.get("fp16", {}).get("enabled") else "fp32"
    zero_stage = ds_config.get("zero_optimization", {}).get("stage")

    train_name = os.path.basename(args.train_data_path)
    dataset_name = os.path.splitext(train_name)[0]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]

    metrics = {
        "schema_version": "2.0",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": safe_get_git_commit(repo_root),
        "command_line": " ".join(sys.argv),
        "env": {
            "python_version": get_python_version(),
            "torch_version": library_versions.get("torch", "unknown"),
            "transformers_version": library_versions.get("transformers", "unknown"),
            "deepspeed_version": library_versions.get("deepspeed", "unknown"),
            "cuda_version": library_versions.get("cuda", "unknown"),
        },
        "host": host_info,
        "hardware": hardware,
        "training_config": {
            "model_name": "gpt2",
            "model_size": f"n_layer={args.num_layers}, n_head={args.num_heads}, n_embd={args.embedding_size}",
            "seq_len": seq_length,
            "epochs": args.epochs,
            "micro_batch_size_per_gpu": micro_batch,
            "grad_accum_steps": grad_accum,
            "global_batch_size": global_batch,
            "lr": optimizer_lr,
            "optimizer": optimizer.get("type"),
            "precision": precision,
            "zero_stage": zero_stage,
            "dataset_name": dataset_name,
            "dataset_subset_sizes": {
                "train": train_sequences,
                "val": val_sequences,
            },
        },
        "deepspeed_config": {
            "path": ds_config_path,
            "sha256": sha256_file(ds_config_path),
            "selected": {
                "zero_stage": zero_stage,
                "fp16_enabled": ds_config.get("fp16", {}).get("enabled"),
                "wall_clock_breakdown": ds_config.get("wall_clock_breakdown"),
            },
        },
        "epochs": [],
        "summary": {
            "total_wall_time_sec": None,
            "best_val_loss": None,
            "mean_tokens_per_sec_global": None,
        },
    }

    return metrics
