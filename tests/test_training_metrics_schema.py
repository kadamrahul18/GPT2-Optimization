import json
from pathlib import Path


def test_training_metrics_schema_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "training_metrics_sample.json"
    with fixture_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "2.0"
    assert isinstance(data["run_id"], str)
    assert isinstance(data["timestamp_utc"], str)
    assert isinstance(data["git_commit"], str)
    assert isinstance(data["command_line"], str)

    env = data["env"]
    assert isinstance(env["python_version"], str)
    assert isinstance(env["torch_version"], str)
    assert isinstance(env["transformers_version"], str)
    assert isinstance(env["deepspeed_version"], str)
    assert isinstance(env["cuda_version"], str)

    hardware = data["hardware"]
    assert hardware["instance_type"] == "g4dn.12xlarge"
    assert isinstance(hardware["gpu_name"], str)
    assert isinstance(hardware["gpu_count_visible"], int)
    assert isinstance(hardware["world_size"], int)

    training_config = data["training_config"]
    assert isinstance(training_config["model_name"], str)
    assert isinstance(training_config["seq_len"], int)
    assert isinstance(training_config["epochs"], int)
    assert isinstance(training_config["micro_batch_size_per_gpu"], int)
    assert isinstance(training_config["grad_accum_steps"], int)
    assert isinstance(training_config["global_batch_size"], int)
    assert isinstance(training_config["lr"], float)
    assert isinstance(training_config["optimizer"], str)
    assert training_config["precision"] in {"fp16_deepspeed", "fp16_torch_amp", "fp32"}
    assert isinstance(training_config["deepspeed_enabled"], bool)
    assert isinstance(training_config["zero_stage"], int)

    epochs = data["epochs"]
    assert isinstance(epochs, list)
    assert len(epochs) > 0
    epoch = epochs[0]
    assert isinstance(epoch["epoch_idx"], int)
    assert isinstance(epoch["epoch_wall_time_sec"], (int, float))
    assert isinstance(epoch["steps"], int)
    assert isinstance(epoch["micro_steps"], int)
    assert isinstance(epoch["optimizer_steps"], int)
    assert isinstance(epoch["tokens_processed_global"], int)
    assert isinstance(epoch["tokens_per_sec_global"], (int, float))
    assert isinstance(epoch["train_loss_avg_global"], (int, float))
    assert isinstance(epoch["val_loss_avg_global"], (int, float))
    assert isinstance(epoch["max_cuda_mem_allocated_bytes_rank0"], int)
    assert isinstance(epoch["max_cuda_mem_reserved_bytes_rank0"], int)
    assert isinstance(epoch["step_time_mean_sec"], (int, float))
    assert isinstance(epoch["step_time_p50_sec"], (int, float))
    assert isinstance(epoch["step_time_p95_sec"], (int, float))
    assert isinstance(epoch["dataload_time_mean_sec"], (int, float))
    completion = data.get("completion")
    assert completion is not None
    assert completion["run_complete_file"] == "RUN_COMPLETE.txt"
    assert isinstance(completion["printed_marker"], bool)
    assert isinstance(completion["timestamp_utc"], str)


def test_slurm_metrics_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "training_metrics_slurm.json"
    with fixture_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "2.0"
    scheduler = data.get("scheduler")
    assert scheduler is not None
    assert scheduler["type"] == "slurm"
    assert data["hardware"]["cluster"] == "Big Purple"


def test_run_complete_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "RUN_COMPLETE.txt"
    content = fixture_path.read_text(encoding="utf-8")
    assert "RUN_COMPLETE" in content
