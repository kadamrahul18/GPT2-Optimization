import argparse
import glob
import json
import os
import re
from datetime import datetime, timezone


def parse_time_percent(line: str):
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
    return float(match.group(1)) if match else None


def parse_total_time_str(line: str):
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(ns|us|ms|s)\b", line)
    if not matches:
        return None
    value, unit = matches[-1]
    return f"{value} {unit}"


def parse_name(line: str):
    # Take everything up to the first numeric field as the "name".
    match = re.search(r"\s+\d", line)
    if not match:
        return None
    return line[: match.start()].strip()


def extract_ranked_items(text: str, kind: str):
    # Heuristic: scan for lines that look like table rows and contain a Time (%) value.
    items = []
    for raw in text.splitlines():
        line = raw.strip("\n")
        if not line or line.startswith(("-", "=")):
            continue
        time_percent = parse_time_percent(line)
        if time_percent is None:
            continue
        name = parse_name(line)
        if not name:
            continue
        total_time = parse_total_time_str(line)
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


def main():
    parser = argparse.ArgumentParser(description="Parse nsys stats text outputs into profile_summary.json")
    parser.add_argument("--run_dir", required=True, help="Run directory containing profiles/")
    args = parser.parse_args()

    profiles_dir = os.path.join(args.run_dir, "profiles")
    stats_files = sorted(glob.glob(os.path.join(profiles_dir, "nsys_stats_*.txt")))
    if not stats_files:
        raise SystemExit(f"No nsys stats files found under: {profiles_dir}")

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

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": os.path.abspath(args.run_dir),
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

