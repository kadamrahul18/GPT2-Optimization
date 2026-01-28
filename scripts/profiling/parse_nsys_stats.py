import argparse
import glob
import json
import os
import re
from datetime import datetime, timezone


def parse_time_percent(value: str):
    value = value.strip()
    if not value:
        return None
    if value.endswith("%"):
        value = value[:-1].strip()
    try:
        return float(value)
    except ValueError:
        return None


def parse_total_time_str(value: str):
    # Nsight Systems commonly prints Total Time as comma-grouped integers under a header like
    # "Total Time (ns)" with no explicit unit per-row. Keep the output as "<int> ns".
    raw = value.strip()
    if not raw:
        return None
    raw = raw.replace(",", "")
    if raw.isdigit():
        return f"{raw} ns"

    # Fallback for other formats that include explicit units.
    match = re.search(r"(\d+(?:\.\d+)?)\s*(ns|us|ms|s)\b", value)
    if not match:
        return None
    number, unit = match.group(1), match.group(2)
    return f"{number} {unit}"


def parse_table_row(line: str):
    # Rows are space-aligned tables where columns are separated by 2+ spaces.
    # The first column is Time (%), and the last column is the name (NVTX range / OSRT symbol).
    columns = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]
    if len(columns) < 3:
        return None

    time_percent = parse_time_percent(columns[0])
    if time_percent is None:
        return None

    total_time = parse_total_time_str(columns[1])
    name = columns[-1]
    if not name:
        return None

    return time_percent, total_time, name


def extract_ranked_items(text: str, kind: str):
    # Scan only the relevant Nsight Systems report section, then parse rows where:
    # - first column is a float time percent
    # - last column is the name (NVTX range / OSRT symbol)
    start_markers = {
        "nvtx": "NVTX Range Summary",
        "osrt": "OS Runtime Summary",
    }
    start_marker = start_markers.get(kind)
    if start_marker is None:
        raise ValueError(f"Unsupported kind: {kind}")

    items = []
    in_section = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if start_marker in stripped:
            in_section = True
            continue
        if in_section and stripped.startswith("**") and start_marker not in stripped:
            in_section = False
        if in_section and stripped.startswith("Processing ["):
            in_section = False
        if not in_section:
            continue

        if not stripped or stripped.startswith(("-", "=")):
            continue

        parsed = parse_table_row(line)
        if parsed is None:
            continue
        time_percent, total_time, name = parsed
        items.append(
            {
                "kind": kind,
                "name": name,
                "time_percent": time_percent,
                "total_time": total_time,
                "raw": line,
            }
        )
    items.sort(key=lambda x: x["time_percent"], reverse=True)
    return items


def read_json_if_exists(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def main():
    parser = argparse.ArgumentParser(description="Parse nsys stats text outputs into profile_summary.json")
    parser.add_argument("--run_dir", required=True, help="Run directory containing profiles/")
    args = parser.parse_args()

    profiles_dir = os.path.join(args.run_dir, "profiles")
    stats_files = sorted(glob.glob(os.path.join(profiles_dir, "nsys_stats_*.txt")))
    if not stats_files:
        raise SystemExit(f"No nsys stats files found under: {profiles_dir}")

    metrics_path = os.path.join(args.run_dir, "training_metrics.json")
    launcher_path = os.path.join(args.run_dir, "launcher_metadata.json")
    training_metrics = read_json_if_exists(metrics_path) or {}
    launcher_metadata = read_json_if_exists(launcher_path) or {}

    nvtx_items = []
    osrt_items = []
    per_file = {}

    for path in stats_files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        nvtx = extract_ranked_items(text, kind="nvtx")
        osrt = extract_ranked_items(text, kind="osrt")

        nvtx_items.extend(nvtx)
        osrt_items.extend(osrt)
        per_file[os.path.basename(path)] = {
            "nvtx_top5": nvtx[:5],
            "osrt_top5": osrt[:5],
        }

    nvtx_items.sort(key=lambda x: x["time_percent"], reverse=True)
    osrt_items.sort(key=lambda x: x["time_percent"], reverse=True)

    nvtx_top5 = nvtx_items[:5]
    osrt_top5 = osrt_items[:5]

    likely_bottleneck = {"allreduce_gradients_over_20pct": False}
    for item in nvtx_items:
        if "allreduce_gradients" in item["name"].lower() and item["time_percent"] > 20.0:
            likely_bottleneck["allreduce_gradients_over_20pct"] = True
            likely_bottleneck["matched"] = {
                "name": item["name"],
                "time_percent": item["time_percent"],
                "total_time": item["total_time"],
            }
            break

    nsys_reports = sorted(
        glob.glob(os.path.join(profiles_dir, "nsys_*.nsys-rep"))
        + glob.glob(os.path.join(profiles_dir, "nsys_*.qdrep"))
    )
    nsys_report_basenames = [os.path.basename(p) for p in nsys_reports]

    scheduler = training_metrics.get("scheduler", {}) if isinstance(training_metrics, dict) else {}
    host_info = training_metrics.get("host", {}) if isinstance(training_metrics, dict) else {}
    hardware = training_metrics.get("hardware", {}) if isinstance(training_metrics, dict) else {}
    env = training_metrics.get("env", {}) if isinstance(training_metrics, dict) else {}
    training_cfg = training_metrics.get("training_config", {}) if isinstance(training_metrics, dict) else {}
    summary = training_metrics.get("summary", {}) if isinstance(training_metrics, dict) else {}

    slurm_meta = launcher_metadata.get("slurm", {}) if isinstance(launcher_metadata, dict) else {}
    slurm_hosts_raw = coalesce(slurm_meta.get("hosts"), launcher_metadata.get("hosts"))
    hosts = None
    if isinstance(slurm_hosts_raw, str) and slurm_hosts_raw.strip():
        hosts = [h for h in slurm_hosts_raw.splitlines() if h.strip()]
    if not hosts and nsys_reports:
        # Derive from filenames: nsys_<jobid>_<host>.*
        derived = []
        for name in nsys_report_basenames:
            m = re.match(r"^nsys_\d+_(.+?)\.(?:nsys-rep|qdrep)$", name)
            if m:
                derived.append(m.group(1))
        if derived:
            hosts = sorted(set(derived))

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": os.path.abspath(args.run_dir),
        "slurm_job_id": coalesce(slurm_meta.get("job_id"), scheduler.get("job_id")),
        "hosts": hosts,
        "world_size": coalesce(launcher_metadata.get("world_size"), hardware.get("world_size")),
        "gpu": hardware.get("gpu_name"),
        "cuda": env.get("cuda_version"),
        "torch": env.get("torch_version"),
        "deepspeed": env.get("deepspeed_version"),
        "seq_length": training_cfg.get("seq_len"),
        "micro_batch_size_per_gpu": training_cfg.get("micro_batch_size_per_gpu"),
        "grad_accum_steps": training_cfg.get("grad_accum_steps"),
        "tokens_per_sec": summary.get("mean_tokens_per_sec_global"),
        "total_wall_time_sec": summary.get("total_wall_time_sec"),
        "nsys_report": nsys_report_basenames[0] if len(nsys_report_basenames) == 1 else nsys_report_basenames,
        "inputs": [os.path.basename(p) for p in stats_files],
        "nvtx_top5": nvtx_top5,
        "osrt_top5": osrt_top5,
        "likely_bottleneck": likely_bottleneck,
        "per_file": per_file,
    }

    out_path = os.path.join(profiles_dir, "profile_summary.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
