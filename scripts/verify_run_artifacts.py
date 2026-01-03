#!/usr/bin/env python3
import argparse
import json
import os
import sys


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "timestamp_utc",
    "git_commit",
    "command_line",
    "env",
    "hardware",
    "training_config",
    "deepspeed_config",
    "epochs",
    "summary",
}


def collect_run_dirs(base_dir):
    run_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        for name in dirs:
            if name.startswith("run_"):
                run_dirs.append(os.path.join(root, name))
        dirs[:] = []
    return sorted(run_dirs)


def expected_world_size(run_dir):
    parts = os.path.normpath(run_dir).split(os.sep)
    for part in parts:
        if part.startswith("1gpu_"):
            return 1
        if part.startswith("2gpu_"):
            return 2
        if part.startswith("4gpu_"):
            return 4
    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_metrics(metrics, expected_ws):
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in metrics]
    world_size = metrics.get("hardware", {}).get("world_size")
    world_ok = expected_ws is None or world_size == expected_ws
    return missing, world_ok, world_size


def main():
    parser = argparse.ArgumentParser(description="Verify training run artifacts.")
    parser.add_argument("--dir", required=True, help="Base run directory")
    args = parser.parse_args()

    run_dirs = collect_run_dirs(args.dir)
    if not run_dirs:
        print("No run directories found.")
        sys.exit(1)

    failures = 0
    rows = []

    for run_dir in run_dirs:
        metrics_path = os.path.join(run_dir, "training_metrics.json")
        log_path = os.path.join(run_dir, "train.log")
        expected_ws = expected_world_size(run_dir)

        metrics_ok = os.path.exists(metrics_path)
        log_ok = os.path.exists(log_path) and os.path.getsize(log_path) > 0

        missing_keys = []
        world_ok = False
        world_size = None
        if metrics_ok:
            try:
                metrics = load_json(metrics_path)
                missing_keys, world_ok, world_size = validate_metrics(metrics, expected_ws)
            except Exception:
                missing_keys = ["invalid_json"]
                world_ok = False

        status = "ok"
        if not metrics_ok:
            status = "missing_metrics"
        elif not log_ok:
            status = "missing_log"
        elif missing_keys:
            status = "invalid_schema"
        elif not world_ok:
            status = "world_size_mismatch"

        if status != "ok":
            failures += 1

        rows.append(
            {
                "run_dir": run_dir,
                "status": status,
                "world_size": world_size,
                "expected_world_size": expected_ws,
            }
        )

    print("run_dir\tstatus\tworld_size\texpected_world_size")
    for row in rows:
        print(
            f"{row['run_dir']}\t{row['status']}\t{row['world_size']}\t{row['expected_world_size']}"
        )

    if failures:
        print(f"Verification failed: {failures} run(s) invalid.")
        sys.exit(1)
    print("Verification passed.")


if __name__ == "__main__":
    main()
