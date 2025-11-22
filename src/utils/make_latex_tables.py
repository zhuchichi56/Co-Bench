import json
import os
from typing import Dict, List, Tuple, Any


# Columns to appear in LaTeX tables (left to right)
BASE_COLUMNS = [
    ("magpie_5k_test", "Magpie"),
    ("numina_cot_5k_test", "Numina"),
    ("mmlu_test", "MMLU"),
    ("__AVG_BASE__", "\\textbf{AVG}"),
    ("mt-bench", "MT-Bench"),
    ("math", "MATH"),
]

MMLU_PRO_SUBJECTS = [
    "biology", "chemistry", "computer_science", "engineering", "math", "physics",
    "history", "philosophy",
    "economics", "law", "psychology",
    "business", "health", "other",
]

CATEGORIES = {
    "STEM": ["biology", "chemistry", "computer_science", "engineering", "math", "physics"],
    "Humanities": ["history", "philosophy"],
    "Social Science": ["economics", "law", "psychology"],
    "Other": ["business", "health", "other"],
}

CATEGORY_COLUMNS = [
    ("STEM", "STEM"),
    ("Humanities", "Humanities"),
    ("Social Science", "Social Sciences"),
    ("Other", "Others"),
    ("__AVG_CAT__", "\\textbf{AVG}"),
]


def _safe_get_metric(item: Dict[str, Any], dataset: str, metric_suffix: str) -> float:
    """Return metric value if present, else None.

    Expects keys like f"{dataset}_{metric_suffix}".
    """
    if not isinstance(item, dict):
        return None
    key = f"{dataset}_{metric_suffix}"
    val = item.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _mean(values: List[float]) -> float:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _format_cell(value: float) -> str:
    if value is None:
        return "--"
    # Default to 3 decimals
    return f"{value:.3f}"


def _collect_dataset_keys_for_categories() -> List[str]:
    keys = []
    for subject in MMLU_PRO_SUBJECTS:
        keys.append(f"mmlu_pro_{subject}")
    return keys


def _compute_category_metric(method_item: Dict[str, Any], metric_suffix: str) -> Dict[str, float]:
    """Compute category-level averages over MMLU-Pro for a given method and metric suffix.

    metric_suffix examples: 'auroc', 'avg_router_score', 'lpm', 'mpm', 'hpm'
    """
    subject_to_value = {}
    for subject in MMLU_PRO_SUBJECTS:
        ds_name = f"mmlu_pro_{subject}"
        subject_to_value[subject] = _safe_get_metric(method_item, ds_name, metric_suffix)

    cat_avgs = {}
    for cat_name, subjects in CATEGORIES.items():
        vals = [subject_to_value.get(s) for s in subjects]
        cat_avgs[cat_name] = _mean(vals)
    return cat_avgs


def _row_for_method(method_name: str, method_item: Dict[str, Any], metric_suffix: str) -> Tuple[str, List[str]]:
    """Build one LaTeX row for a method for the specified metric suffix.

    Returns: (row_leading_cell, cells) where cells align to BASE_COLUMNS + CATEGORY_COLUMNS
    """
    # Base datasets
    base_vals: List[float] = []
    for key, _label in BASE_COLUMNS:
        if key == "__AVG_BASE__":
            base_vals.append(_mean(base_vals))
        else:
            base_vals.append(_safe_get_metric(method_item, key, metric_suffix))

    # MMLU-Pro categories
    cat_metrics = _compute_category_metric(method_item, metric_suffix)
    cat_vals: List[float] = []
    for key, _label in CATEGORY_COLUMNS:
        if key == "__AVG_CAT__":
            cat_vals.append(_mean([cat_metrics.get(k) for k in CATEGORIES.keys()]))
        else:
            cat_vals.append(cat_metrics.get(key))

    # Format all cells
    cells = [_format_cell(v) for v in base_vals + cat_vals]
    return method_name, cells


def build_latex_table(results: Dict[str, Any], metric: str) -> str:
    """Create LaTeX table body (rows only) for a given metric.

    Supported metrics mapping:
      - 'auroc' -> uses per-dataset key '{dataset}_auroc'
      - 'lpm'   -> expects '{dataset}_lpm' if present, else falls back to '{dataset}_avg_router_score'
      - 'mpm'   -> expects '{dataset}_mpm' if present
      - 'hpm'   -> expects '{dataset}_hpm' if present
    """
    if metric == "auroc":
        metric_suffix = "auroc"
    elif metric == "lpm":
        metric_suffix = "lpm"
    elif metric == "mpm":
        metric_suffix = "mpm"
    elif metric == "hpm":
        metric_suffix = "hpm"
    else:
        raise ValueError("Unsupported metric: " + metric)

    # If lpm is absent in data, gracefully fallback to avg_router_score
    def metric_or_fallback(item: Dict[str, Any]) -> Dict[str, Any]:
        if metric_suffix != "lpm":
            return item
        # Build a shallow copy with lpm populated when missing
        new_item = dict(item)
        keys = list(item.keys())
        for k in keys:
            if k.endswith("_avg_router_score"):
                ds = k[: -len("_avg_router_score")]
                lpm_key = f"{ds}_lpm"
                if lpm_key not in new_item:
                    new_item[lpm_key] = item[k]
        return new_item

    # Header (user wants: first column method name)
    header_parts = ["Method"]
    header_parts += [label for _key, label in BASE_COLUMNS]
    header_parts += [label for _key, label in CATEGORY_COLUMNS]
    header_line = " & ".join(header_parts) + " \\"  # end row

    lines = [header_line]

    for method_name, method_item in results.items():
        fixed_item = metric_or_fallback(method_item)
        row_name, cells = _row_for_method(method_name, fixed_item, metric_suffix)
        line = row_name + " & " + " & ".join(cells) + " \\"  # end row
        lines.append(line)

    return "\n".join(lines)


def generate_all_tables(results_path: str) -> Dict[str, str]:
    """Load results json and generate LaTeX table strings for auroc, lpm, mpm, hpm."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(results_path)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    tables = {}
    for metric in ["auroc", "lpm", "mpm", "hpm"]:
        try:
            tables[metric] = build_latex_table(results, metric)
        except ValueError:
            # Skip unsupported metric
            continue
    return tables


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to evaluation JSON, e.g., src/logits_based_routers_evaluation.json")
    args = parser.parse_args()

    tables = generate_all_tables(args.results)

    print("\n=== AUROC (copy rows into LaTeX) ===")
    if "auroc" in tables:
        print(tables["auroc"])  # rows only
    else:
        print("No AUROC table available.")

    print("\n=== LPM (copy rows into LaTeX) ===")
    if "lpm" in tables:
        print(tables["lpm"])  # rows only
    else:
        print("No LPM table available.")

    print("\n=== MPM (copy rows into LaTeX) ===")
    if "mpm" in tables:
        print(tables["mpm"])  # rows only
    else:
        print("No MPM table available.")

    print("\n=== HPM (copy rows into LaTeX) ===")
    if "hpm" in tables:
        print(tables["hpm"])  # rows only
    else:
        print("No HPM table available.")


