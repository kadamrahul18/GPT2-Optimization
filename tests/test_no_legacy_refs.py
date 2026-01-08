import os
from pathlib import Path


FORBIDDEN = [
    "baseline_metrics.json",
    "baseline_profiler_logs",
    "kernel-level comparison",
    "Attempting kernel-level comparison",
    ".pt.trace.json",
]


def test_no_legacy_references():
    repo_root = Path(__file__).resolve().parents[1]
    for root, dirs, files in os.walk(repo_root):
        rel = Path(root).relative_to(repo_root)
        if rel.parts and rel.parts[0] in {".git", "checkpoint"}:
            dirs[:] = []
            continue
        if "__pycache__" in rel.parts:
            dirs[:] = []
            continue
        for name in files:
            if name.endswith((".png", ".pdf")):
                continue
            if name.endswith(".pyc"):
                continue
            if name in {"repo_llm_input.txt", "test_no_legacy_refs.py"}:
                continue
            path = Path(root) / name
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for token in FORBIDDEN:
                assert token not in content, f"Found legacy reference '{token}' in {path}"
