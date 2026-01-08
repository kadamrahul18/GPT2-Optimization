#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_metrics(path):
    data = json.loads(Path(path).read_text())
    hw = data.get("hardware", {})
    cfg = data.get("training_config", {})
    epochs = data.get("epochs", [])
    last = epochs[-1] if epochs else {}
    summary = data.get("summary", {})
    return {
        "gpu_name": hw.get("gpu_name"),
        "cluster": hw.get("cluster"),
        "world_size": hw.get("world_size"),
        "seq_len": cfg.get("seq_len"),
        "global_batch_size": cfg.get("global_batch_size"),
        "micro_batch_size_per_gpu": cfg.get("micro_batch_size_per_gpu"),
        "grad_accum_steps": cfg.get("grad_accum_steps"),
        "optimizer_steps": last.get("optimizer_steps"),
        "tokens_processed_global": last.get("tokens_processed_global"),
        "tokens_per_sec": summary.get("mean_tokens_per_sec_global"),
        "total_wall_time_sec": summary.get("total_wall_time_sec"),
        "git_commit": data.get("git_commit"),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate scaling table from training_metrics.json files.")
    parser.add_argument("--gpu1", required=True)
    parser.add_argument("--gpu2", required=True)
    parser.add_argument("--gpu4", required=True)
    args = parser.parse_args()

    m1 = load_metrics(args.gpu1)
    m2 = load_metrics(args.gpu2)
    m4 = load_metrics(args.gpu4)

    base = m1["tokens_per_sec"] or 1.0

    def row(metrics):
        tps = metrics["tokens_per_sec"]
        wall = metrics["total_wall_time_sec"]
        speedup = tps / base if tps else None
        eff = speedup / metrics["world_size"] if speedup else None
        return tps, wall, speedup, eff

    rows = {1: row(m1), 2: row(m2), 4: row(m4)}

    print("| GPUs | Tokens/sec (global) | Wall time (s) | Speedup vs 1 GPU | Scaling efficiency |")
    print("| --- | --- | --- | --- | --- |")
    for g in (1, 2, 4):
        tps, wall, speedup, eff = rows[g]
        print(f"| {g} | {tps:.2f} | {wall:.2f} | {speedup:.2f} | {eff:.2f} |")


if __name__ == "__main__":
    main()
