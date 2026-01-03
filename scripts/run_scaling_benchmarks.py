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
    for part in parts:
        try:
            gpus.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid GPU count: {part}") from exc
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
    stdout_path = os.path.join(log_dir, "train.log")
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


def mean_and_stdev(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


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


def build_scaling_report(metrics_paths, seq_len, dataset_mode, instance_type):
    baseline_metrics = load_metrics(metrics_paths[1][0])
    baseline_throughput = mean_tokens_per_sec(baseline_metrics)
    baseline_epoch_time = mean_epoch_wall_time(baseline_metrics)

    report = {
        "status": "ok",
        "protocol": {
            "instance_type": instance_type,
            "seq_len": seq_len,
            "dataset_mode": dataset_mode,
            "training_config": baseline_metrics.get("training_config", {}),
        },
        "runs": {},
    }

    for gpu_count in sorted(metrics_paths.keys()):
        metrics_list = [load_metrics(p) for p in metrics_paths[gpu_count]]
        throughputs = [mean_tokens_per_sec(m) for m in metrics_list if mean_tokens_per_sec(m) is not None]
        epoch_times = [mean_epoch_wall_time(m) for m in metrics_list if mean_epoch_wall_time(m) is not None]

        throughput_mean, throughput_stdev = mean_and_stdev(throughputs)
        epoch_time_mean, epoch_time_stdev = mean_and_stdev(epoch_times)

        speedup, efficiency = compute_speedup(
            throughput_mean,
            baseline_throughput,
            epoch_time_mean,
            baseline_epoch_time,
            gpu_count,
        )

        if gpu_count == 1:
            speedup = 1.0 if throughput_mean is not None or epoch_time_mean is not None else None
            efficiency = 1.0 if throughput_mean is not None or epoch_time_mean is not None else None

        report["runs"][str(gpu_count)] = {
            "metrics_paths": metrics_paths[gpu_count],
            "mean_tokens_per_sec_global": throughput_mean,
            "stdev_tokens_per_sec_global": throughput_stdev,
            "mean_epoch_wall_time_sec": epoch_time_mean,
            "stdev_epoch_wall_time_sec": epoch_time_stdev,
            "speedup_vs_1gpu": speedup,
            "scaling_efficiency": efficiency,
        }

    return report


def validate_invariants(metrics_paths):
    baseline = load_metrics(metrics_paths[1][0])
    baseline_cfg = baseline.get("training_config", {})
    baseline_dataset = baseline_cfg.get("dataset_subset_sizes", {})

    mismatches = []

    def compare_field(name, other_value, baseline_value):
        if other_value != baseline_value:
            mismatches.append(
                {
                    "field": name,
                    "baseline": baseline_value,
                    "mismatch": other_value,
                }
            )

    for gpu_count, paths in metrics_paths.items():
        for path in paths:
            metrics = load_metrics(path)
            cfg = metrics.get("training_config", {})
            dataset = cfg.get("dataset_subset_sizes", {})

            compare_field("seq_len", cfg.get("seq_len"), baseline_cfg.get("seq_len"))
            compare_field("dataset_name", cfg.get("dataset_name"), baseline_cfg.get("dataset_name"))
            compare_field("dataset_subset_sizes", dataset, baseline_dataset)
            compare_field("global_batch_size", cfg.get("global_batch_size"), baseline_cfg.get("global_batch_size"))
            compare_field("epochs", cfg.get("epochs"), baseline_cfg.get("epochs"))
            compare_field("model_size", cfg.get("model_size"), baseline_cfg.get("model_size"))
            compare_field("precision", cfg.get("precision"), baseline_cfg.get("precision"))

    return mismatches


def build_invalid_report(mismatches):
    return {
        "status": "invalid_comparison",
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description="Run scaling benchmarks on a single machine.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: checkpoint/scaling_runs/<timestamp>)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--dataset_mode", default="small", choices=["small", "full"])
    parser.add_argument("--deepspeed_config", default="src/deepspeed_config.json")
    parser.add_argument("--gpus", type=parse_gpu_list, default=[1, 2, 4])
    parser.add_argument("--repeat", type=int, default=1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_slurm_tmpdir", dest="use_slurm_tmpdir", action="store_true")
    group.add_argument("--no_use_slurm_tmpdir", dest="use_slurm_tmpdir", action="store_false")
    parser.set_defaults(use_slurm_tmpdir=True)
    parser.add_argument("--allow_mismatch", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or os.path.join("checkpoint", "scaling_runs", timestamp)

    use_slurm_tmpdir = args.use_slurm_tmpdir and os.getenv("SLURM_TMPDIR")
    tmp_base_dir = os.getenv("SLURM_TMPDIR") if use_slurm_tmpdir else None
    run_base_dir = tmp_base_dir if tmp_base_dir else out_dir

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

        for run_idx in range(1, args.repeat + 1):
            run_root = os.path.join(run_base_dir, run_name, f"run_{run_idx}")
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
            run_configs.append((gpu_count, run_name, run_root, run_idx, cmd, env))

    metrics_paths = {1: [], 2: [], 4: []}
    for gpu_count, run_name, run_root, run_idx, cmd, env in run_configs:
        print(f"Running {run_name} run_{run_idx}...")
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
            metrics_paths[gpu_count].append(metrics_path)
            continue
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")

        if use_slurm_tmpdir:
            final_dir = os.path.join(out_dir, run_name, f"run_{run_idx}")
            ensure_dir(os.path.dirname(final_dir))
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            shutil.copytree(run_root, final_dir)
            metrics_paths[gpu_count].append(os.path.join(final_dir, "training_metrics.json"))
        else:
            metrics_paths[gpu_count].append(metrics_path)

    if args.dry_run:
        print("Dry run complete. No report generated.")
        print(f"Resolved output directory: {out_dir}")
        if use_slurm_tmpdir:
            print(f"Resolved SLURM_TMPDIR base: {tmp_base_dir}")
        return

    mismatches = validate_invariants(metrics_paths)
    if mismatches and not args.allow_mismatch:
        report = build_invalid_report(mismatches)
        report_path = os.path.join(out_dir, "scaling_report.json")
        ensure_dir(out_dir)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        raise SystemExit("Invariant check failed; comparison invalid. See scaling_report.json for details.")

    instance_type = os.getenv("EC2_INSTANCE_TYPE") or "unknown"
    report = build_scaling_report(
        metrics_paths=metrics_paths,
        seq_len=args.seq_len,
        dataset_mode=args.dataset_mode,
        instance_type=instance_type,
    )
    if mismatches and args.allow_mismatch:
        report["status"] = "mismatch_allowed"
        report["mismatches"] = mismatches

    report_path = os.path.join(out_dir, "scaling_report.json")
    ensure_dir(out_dir)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Scaling report written to {report_path}")


if __name__ == "__main__":
    main()
