#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime


def parse_gpu_list(value):
    parts = [p.strip() for p in value.split(",") if p.strip()]
    gpus = []
    for p in parts:
        try:
            gpus.append(int(p))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid GPU count: {p}") from exc
    return gpus


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dataset_paths(dataset_mode):
    if dataset_mode == "small":
        return "train_small.bin", "val_small.bin"
    if dataset_mode == "full":
        return "train.bin", "val.bin"
    raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")


def build_command(
    gpu_count,
    deepspeed_config,
    checkpoint_path,
    epochs,
    seq_len,
    train_path,
    val_path,
    run_type,
):
    base_cmd = [
        sys.executable,
        "src/gpt2.py",
        "--run_type",
        run_type,
        "--train_data_path",
        train_path,
        "--val_data_path",
        val_path,
        "--checkpoint_path",
        checkpoint_path,
        "--epochs",
        str(epochs),
        "--seq_length",
        str(seq_len),
        "--deepspeed_config",
        deepspeed_config,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))

    if gpu_count == 1:
        return base_cmd, env

    ds_cmd = ["deepspeed", "src/gpt2.py"]
    ds_cmd += base_cmd[2:]
    return ds_cmd, env


def run_command(cmd, env, workdir, log_dir, dry_run):
    ensure_dir(log_dir)
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0, stdout_path, stderr_path

    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        result = subprocess.run(cmd, cwd=workdir, env=env, stdout=out, stderr=err)
    return result.returncode, stdout_path, stderr_path


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_epoch_wall_time(metrics):
    epochs = metrics.get("epochs", [])
    if not epochs:
        return None
    times = [e.get("epoch_wall_time_sec") for e in epochs if e.get("epoch_wall_time_sec") is not None]
    if not times:
        return None
    return sum(times) / len(times)


def mean_tokens_per_sec(metrics):
    summary = metrics.get("summary", {})
    value = summary.get("mean_tokens_per_sec_global")
    if value is not None:
        return value
    epochs = metrics.get("epochs", [])
    if not epochs:
        return None
    values = [e.get("tokens_per_sec_global") for e in epochs if e.get("tokens_per_sec_global") is not None]
    if not values:
        return None
    return sum(values) / len(values)


def compute_speedup(throughput, baseline_throughput, wall_time, baseline_wall_time, gpu_count):
    if throughput is not None and baseline_throughput not in (None, 0):
        speedup = throughput / baseline_throughput
        efficiency = speedup / gpu_count
        return speedup, efficiency

    if wall_time not in (None, 0) and baseline_wall_time not in (None, 0):
        speedup = baseline_wall_time / wall_time
        efficiency = speedup / gpu_count
        return speedup, efficiency

    return None, None


def build_scaling_report(metrics_paths, seq_len, dataset_mode, instance_type="g4dn.12xlarge"):
    baseline_metrics = load_metrics(metrics_paths[1])
    baseline_throughput = mean_tokens_per_sec(baseline_metrics)
    baseline_epoch_time = mean_epoch_wall_time(baseline_metrics)

    report = {
        "protocol": {
            "instance_type": instance_type,
            "seq_len": seq_len,
            "dataset_mode": dataset_mode,
            "training_config": baseline_metrics.get("training_config", {}),
        },
        "runs": {},
    }

    for gpu_count in sorted(metrics_paths.keys()):
        metrics = load_metrics(metrics_paths[gpu_count])
        throughput = mean_tokens_per_sec(metrics)
        epoch_time = mean_epoch_wall_time(metrics)
        speedup, efficiency = compute_speedup(
            throughput,
            baseline_throughput,
            epoch_time,
            baseline_epoch_time,
            gpu_count,
        )

        if gpu_count == 1:
            speedup = 1.0 if throughput is not None or epoch_time is not None else None
            efficiency = 1.0 if throughput is not None or epoch_time is not None else None

        report["runs"][str(gpu_count)] = {
            "metrics_path": metrics_paths[gpu_count],
            "mean_tokens_per_sec_global": throughput,
            "mean_epoch_wall_time_sec": epoch_time,
            "speedup_vs_1gpu": speedup,
            "scaling_efficiency": efficiency,
        }

    return report


def main():
    parser = argparse.ArgumentParser(description="Run scaling benchmarks on a single machine.")
    parser.add_argument("--base_dir", default="checkpoint/scaling_runs", help="Base output directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--dataset_mode", default="small", choices=["small", "full"])
    parser.add_argument("--deepspeed_config", default="src/deepspeed_config.json")
    parser.add_argument("--gpus", type=parse_gpu_list, default=[1, 2, 4])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path, val_path = dataset_paths(args.dataset_mode)

    run_configs = []
    for gpu_count in args.gpus:
        if gpu_count == 1:
            run_name = "1gpu_baseline"
            run_type = "baseline"
        elif gpu_count == 2:
            run_name = "2gpu_deepspeed"
            run_type = "optimized"
        elif gpu_count == 4:
            run_name = "4gpu_deepspeed"
            run_type = "optimized"
        else:
            raise ValueError("Supported GPU counts are 1, 2, and 4.")

        run_root = os.path.join(args.base_dir, run_name, "run_tmp")
        ensure_dir(run_root)

        cmd, env = build_command(
            gpu_count=gpu_count,
            deepspeed_config=args.deepspeed_config,
            checkpoint_path=run_root,
            epochs=args.epochs,
            seq_len=args.seq_len,
            train_path=train_path,
            val_path=val_path,
            run_type=run_type,
        )
        run_configs.append((gpu_count, run_name, run_root, cmd, env))

    metrics_paths = {}
    for gpu_count, run_name, run_root, cmd, env in run_configs:
        print(f"Running {run_name}...")
        returncode, stdout_path, stderr_path = run_command(
            cmd, env, repo_root, run_root, args.dry_run
        )
        if returncode != 0:
            raise RuntimeError(
                f"{run_name} failed with exit code {returncode}. "
                f"See logs: {stdout_path}, {stderr_path}"
            )

        metrics_path = os.path.join(run_root, "training_metrics.json")
        if args.dry_run:
            metrics_paths[gpu_count] = metrics_path
            continue
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")

        metrics = load_metrics(metrics_path)
        run_id = metrics.get("run_id", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
        final_dir = os.path.join(args.base_dir, run_name, run_id)
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.move(run_root, final_dir)
        metrics_paths[gpu_count] = os.path.join(final_dir, "training_metrics.json")

    if args.dry_run:
        print("Dry run complete. No report generated.")
        return

    report = build_scaling_report(
        metrics_paths=metrics_paths,
        seq_len=args.seq_len,
        dataset_mode=args.dataset_mode,
        instance_type="g4dn.12xlarge",
    )
    report_path = os.path.join(args.base_dir, "scaling_report.json")
    ensure_dir(args.base_dir)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Scaling report written to {report_path}")


if __name__ == "__main__":
    main()
