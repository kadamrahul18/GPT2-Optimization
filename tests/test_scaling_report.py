import json
from pathlib import Path
import importlib.util


def load_runner_module():
    repo_root = Path(__file__).resolve().parents[1]
    runner_path = repo_root / "scripts" / "run_scaling_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_scaling_benchmarks", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_metrics(path, throughput, epoch_time):
    metrics = {
        "schema_version": "2.0",
        "run_id": path.parent.name,
        "training_config": {
            "seq_len": 512,
            "dataset_subset_sizes": {"train": 100, "val": 20},
        },
        "summary": {
            "mean_tokens_per_sec_global": throughput,
            "total_wall_time_sec": epoch_time,
            "best_val_loss": 2.0,
        },
        "epochs": [
            {
                "epoch_idx": 1,
                "epoch_wall_time_sec": epoch_time,
                "tokens_per_sec_global": throughput,
                "tokens_processed_global": 1000,
                "steps": 10,
                "train_loss_avg_global": 2.0,
                "val_loss_avg_global": 2.1,
                "max_cuda_mem_allocated_bytes_rank0": 1,
                "max_cuda_mem_reserved_bytes_rank0": 2,
                "step_time_mean_sec": 0.1,
                "step_time_p50_sec": 0.1,
                "step_time_p95_sec": 0.1,
                "dataload_time_mean_sec": 0.01,
            }
        ],
    }
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_scaling_report_computation(tmp_path):
    module = load_runner_module()

    base = tmp_path / "scaling_runs"
    paths = {
        1: base / "1gpu" / "training_metrics.json",
        2: base / "2gpu" / "training_metrics.json",
        4: base / "4gpu" / "training_metrics.json",
    }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    write_metrics(paths[1], throughput=100.0, epoch_time=10.0)
    write_metrics(paths[2], throughput=180.0, epoch_time=6.0)
    write_metrics(paths[4], throughput=320.0, epoch_time=3.5)

    report = module.build_scaling_report(
        metrics_paths={1: [str(paths[1])], 2: [str(paths[2])], 4: [str(paths[4])]},
        seq_len=512,
        dataset_mode="small",
        instance_type="g4dn.12xlarge",
    )

    runs = report["runs"]
    assert report["status"] == "ok"
    assert runs["1"]["speedup_vs_1gpu"] == 1.0
    assert runs["1"]["scaling_efficiency"] == 1.0

    assert abs(runs["2"]["speedup_vs_1gpu"] - 1.8) < 1e-6
    assert abs(runs["2"]["scaling_efficiency"] - 0.9) < 1e-6
    assert runs["2"]["stdev_tokens_per_sec_global"] == 0.0
    assert runs["2"]["stdev_epoch_wall_time_sec"] == 0.0

    assert abs(runs["4"]["speedup_vs_1gpu"] - 3.2) < 1e-6
    assert abs(runs["4"]["scaling_efficiency"] - 0.8) < 1e-6
    assert runs["4"]["stdev_tokens_per_sec_global"] == 0.0
    assert runs["4"]["stdev_epoch_wall_time_sec"] == 0.0


def test_scaling_report_repeats_from_fixtures():
    module = load_runner_module()
    fixtures = Path(__file__).parent / "fixtures" / "repeat"
    metrics_paths = {
        1: [
            str(fixtures / "1gpu_run1.json"),
            str(fixtures / "1gpu_run2.json"),
            str(fixtures / "1gpu_run3.json"),
        ],
        2: [
            str(fixtures / "2gpu_run1.json"),
            str(fixtures / "2gpu_run2.json"),
            str(fixtures / "2gpu_run3.json"),
        ],
        4: [
            str(fixtures / "4gpu_run1.json"),
            str(fixtures / "4gpu_run2.json"),
            str(fixtures / "4gpu_run3.json"),
        ],
    }

    report = module.build_scaling_report(
        metrics_paths=metrics_paths,
        seq_len=512,
        dataset_mode="small",
        instance_type="g4dn.12xlarge",
    )

    runs = report["runs"]
    assert abs(runs["1"]["mean_tokens_per_sec_global"] - 100.0) < 1e-6
    assert abs(runs["1"]["stdev_tokens_per_sec_global"] - 5.0) < 1e-6
    assert abs(runs["4"]["mean_tokens_per_sec_global"] - 320.0) < 1e-6
    assert abs(runs["4"]["stdev_tokens_per_sec_global"] - 10.0) < 1e-6
    assert runs["2"]["stdev_tokens_per_sec_global"] > 0.0


def test_invariant_mismatch_report():
    module = load_runner_module()
    fixtures = Path(__file__).parent / "fixtures" / "mismatch"
    metrics_paths = {
        1: [str(fixtures / "baseline.json")],
        2: [str(fixtures / "other.json")],
        4: [str(fixtures / "baseline.json")],
    }
    mismatches = module.validate_invariants(metrics_paths)
    assert mismatches
    report = module.build_invalid_report(mismatches)
    assert report["status"] == "invalid_comparison"
    assert report["mismatches"]
