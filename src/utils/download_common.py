"""
Shared helpers for dataset download/convert scripts in src/utils/.

Goal: keep individual download_*.py scripts tiny and consistent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def require_hf_datasets() -> None:
    """Fail fast with a clear message if HuggingFace datasets is missing."""
    try:
        import datasets  # noqa: F401
    except Exception:
        print("error: missing dependency 'datasets'. Install with: pip install datasets")
        sys.exit(1)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def head(dataset: Any, n: int) -> List[Any]:
    """Return first n items from a datasets.Dataset-like object."""
    n = max(0, int(n))
    if n == 0:
        return []
    out = []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        out.append(item)
    return out


def ensure_int(name: str, v: Any, min_value: int = 1) -> int:
    try:
        v = int(v)
    except Exception:
        raise ValueError(f"{name} must be an integer")
    if v < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return v


def get_output_data_dir() -> Path:
    """Standard output location: <repo>/src/data/"""
    return Path(__file__).resolve().parent.parent / "data"


def human_mb(num_bytes: int) -> str:
    try:
        return f"{(float(num_bytes) / 1024 / 1024):.2f} MB"
    except Exception:
        return "n/a"


def maybe_print(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def print_preview(
    rows: List[Dict[str, Any]],
    n: int = 3,
    instruction_chars: int = 150,
    response_chars: int = 100,
) -> None:
    n = max(0, int(n))
    if n == 0 or not rows:
        return
    print("\nPreview:")
    for i, row in enumerate(rows[:n], 1):
        inst = str(row.get("instruction", ""))
        resp = str(row.get("response", ""))
        inst_short = inst[:instruction_chars] + ("..." if len(inst) > instruction_chars else "")
        resp_short = resp[:response_chars] + ("..." if len(resp) > response_chars else "")
        print(f"\n[{i}]")
        print(f"  instruction: {inst_short}")
        print(f"  response: {resp_short}")

