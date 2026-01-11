import json
import os
os.environ["WANDB_API_KEY"] = "79a88980fe13540412ac35e9673ca1ebe5e23380"
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def extract_method_from_dirname(dirname):
    """Extract method name from a directory name.

    Examples:
        alpaca_5k_test_coe -> coe
        mmlu_pro_biology_entropy -> entropy
    """
    known_methods = [
        "coe", "entropy", "confidence_margin", "max_logits",
        "top10_variance", "semantic_entropy", "self_questioning",
        "dirichlet", "probe", "dynamic", "dynamic_fusion_sampling", "embedding_mlp",
        "trained_deberta"
    ]

    for method in known_methods:
        if dirname.endswith(f"_{method}"):
            return method

    return None

def extract_dataset_from_dirname(dirname, method):
    """Extract dataset name from a directory name.

    Examples:
        alpaca_5k_test_coe (method=coe) -> alpaca_5k_test
        mmlu_pro_biology_entropy (method=entropy) -> mmlu_pro_biology
    """
    if method and dirname.endswith(f"_{method}"):
        return dirname[:-len(f"_{method}")]
    return dirname

def load_individual_datasets(method_dir, method_name):
    """Load per-dataset `metrics.json` under a method directory.

    Args:
        method_dir: method directory path, e.g. .../base/coe
        method_name: method name, e.g. coe
    """

    # Standard dataset names
    target_datasets = ["alpaca_5k_test", "big_math_5k_test", "mmlu_test", "magpie_5k_test", "math"]

    results = {}

    # Traverse dataset subdirectories under the method directory
    for item in os.listdir(method_dir):
        item_path = os.path.join(method_dir, item)
        if not os.path.isdir(item_path):
            continue

        # Extract dataset name
        dataset_name = extract_dataset_from_dirname(item, method_name)

        # Only keep target datasets (exclude mmlu_pro here)
        if dataset_name not in target_datasets:
            continue

        metrics_file = os.path.join(item_path, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                metrics = {
                    "auroc": data["reliable_metrics"]["auroc"],
                    "LPM": data["adaptive_metrics"]["LPM"],
                    "HPM": data["adaptive_metrics"]["HPM"],
                    "MPM": data["adaptive_metrics"]["MPM"]
                }

                # For alpaca/magpie, scale LPM/MPM by 1/10 (HPM unchanged)
                if 'alpaca' in dataset_name.lower() or 'magpie' in dataset_name.lower():
                    
                    metrics["LPM"] = metrics["LPM"] / 10.0 if metrics["LPM"] is not None else None
                    metrics["MPM"] = metrics["MPM"] / 10.0 if metrics["MPM"] is not None else None
                    # HPM unchanged

                results[dataset_name] = metrics

            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
                results[dataset_name] = None
        else:
            print(f"Metrics file not found: {metrics_file}")
            results[dataset_name] = None

    # Fill missing datasets with None
    for dataset in target_datasets:
        if dataset not in results:
            results[dataset] = None

    return results

def load_mmlu_pro_categories(method_dir, method_name):
    """Load MMLU-Pro category-level metrics.

    Args:
        method_dir: method directory path, e.g. .../base/coe
        method_name: method name, e.g. coe
    """

    categories = {
        "STEM": ["biology", "chemistry", "computer_science", "engineering", "math", "physics"],
        "Humanities": ["history", "philosophy"],
        "Social Science": ["economics", "law", "psychology"],
        "Other": ["business", "health", "other"]
    }

    category_results = defaultdict(list)

    for item in os.listdir(method_dir):
        item_path = os.path.join(method_dir, item)
        if not os.path.isdir(item_path):
            continue

        # Extract dataset name
        dataset_name = extract_dataset_from_dirname(item, method_name)

        # Only process mmlu_pro_*
        if not dataset_name.startswith("mmlu_pro_"):
            continue

        # Extract subject
        subject = dataset_name.replace("mmlu_pro_", "")

        # Map subject to category
        category = None
        for cat, subjects in categories.items():
            if subject in subjects:
                category = cat
                break

        if category is None:
            print(f"Warning: Subject '{subject}' not found in any category")
            continue

        metrics_file = os.path.join(item_path, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                metrics = {
                    "subject": subject,
                    "auroc": data["reliable_metrics"]["auroc"],
                    "LPM": data["adaptive_metrics"]["LPM"],
                    "HPM": data["adaptive_metrics"]["HPM"],
                    "MPM": data["adaptive_metrics"]["MPM"]
                }

                category_results[category].append(metrics)

            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")

    # Compute per-category averages
    averages = {}
    category_order = ["STEM", "Humanities", "Social Science", "Other"]

    for category in category_order:
        if category in category_results and category_results[category]:
            metrics_list = category_results[category]
            averages[category] = {
                "auroc": sum(m["auroc"] for m in metrics_list) / len(metrics_list),
                "LPM": sum(m["LPM"] for m in metrics_list) / len(metrics_list),
                "HPM": sum(m["HPM"] for m in metrics_list) / len(metrics_list),
                "MPM": sum(m["MPM"] for m in metrics_list) / len(metrics_list)
            }
        else:
            averages[category] = None

    return averages


def print_metric_comparison_table(all_path_results, metric, method_names, highlight_methods=None, print_header=True, print_footer=True):
    """Print a formatted comparison table for a given metric."""

    dataset_order = ["alpaca_5k_test", "big_math_5k_test", "mmlu_test", "magpie_5k_test", "math"]
    dataset_display_names = ["Alpaca", "Numina", "MMLU", "Magpie", "MATH"]

    first_three_keys = dataset_order[:3]
    first_three_names = dataset_display_names[:3]
    last_two_keys = dataset_order[3:]
    last_two_names = dataset_display_names[3:]

    mmlu_pro_order = ["STEM", "Humanities", "Social Science", "Other"]

    # Method grouping map
    method_groups = {
        # Verb-based methods
        "semantic_entropy": ("Verb", "SemanticEntropy"),
        "self_questioning": ("Verb", "SelfAsk"),
        # Logit-based methods
        "confidence_margin": ("Logit", "ConfidenceMargin"),
        "entropy": ("Logit", "Entropy"),
        "max_logits": ("Logit", "MaxLogits"),
        "top10_variance": ("Logit", "Top10Variance"),
        # Probe-based methods
        # "coe": ("Probe", "ProbeCoE"),
        # "coe_dual": ("Probe", "ProbeCoE"),
        # "hs_last": ("Probe", "ProbeFinal"),
        # "mean": ("Probe", "ProbeMean"),
        "probe": ("Probe", "ProbeMean"),
        "dynamic": ("Probe", "ProbeDirichlet"),
        "dirichlet": ("Probe", "Dirichlet"),
        "dynamic_fusion_sampling": ("Probe", "DirichletSampling"),
        "embedding_mlp": ("Embedding", "EmbeddingMLP"),
        "deberta": ("Embedding", "DeBERTa"),
        "trained_deberta": ("Embedding", "DeBERTa")

    }

    def fmt(x):
        """Format a ratio in [0,1] as a percentage with 2 decimals; None -> '-'."""
        if x is None:
            return "-"
        return f"{x * 100:.2f}"

    if print_header:
        print(f"\n{metric.upper()} Comparison Table:")
        print("=" * 200)

        all_columns = first_three_names + ["Avg(1-3)"] + last_two_names + mmlu_pro_order + ["Avg(4-9)"]
        header = "Method & " + " & ".join([f"\\textbf{{{col}}}" for col in all_columns]) + " \\\\"
        print(header)
        print("\\hline")

    # Collect all rows to compute per-column maxima
    all_data = []
    row_count = min(len(method_names), len(all_path_results))

    for method_idx in range(row_count):
        individual_results, mmlu_pro_averages = all_path_results[method_idx]

        row_data = []

        # First three datasets
        first_three_vals = []
        for ds in first_three_keys:
            v = None if individual_results.get(ds) is None else individual_results[ds].get(metric, None)
            row_data.append(v)
            if v is not None:
                first_three_vals.append(v)

        # Avg(1-3)
        if first_three_vals:
            avg_1_3 = sum(first_three_vals) / len(first_three_vals)
            row_data.append(avg_1_3)
        else:
            row_data.append(None)

        # Last two datasets
        last_two_vals = []
        for ds in last_two_keys:
            v = None if individual_results.get(ds) is None else individual_results[ds].get(metric, None)
            row_data.append(v)
            if v is not None:
                last_two_vals.append(v)

        # MMLU-Pro categories
        pro_vals = []
        for cat in mmlu_pro_order:
            v = None if mmlu_pro_averages.get(cat) is None else mmlu_pro_averages[cat].get(metric, None)
            row_data.append(v)
            if v is not None:
                pro_vals.append(v)

        # Avg(4-9)
        avg_4_9_pool = last_two_vals + pro_vals
        if avg_4_9_pool:
            avg_4_9 = sum(avg_4_9_pool) / len(avg_4_9_pool)
            row_data.append(avg_4_9)
        else:
            row_data.append(None)

        all_data.append(row_data)

    # Max value per column
    num_cols = len(all_data[0]) if all_data else 0
    max_indices = []
    for col_idx in range(num_cols):
        col_values = [row[col_idx] for row in all_data if row[col_idx] is not None]
        if col_values:
            max_val = max(col_values)
            max_indices.append(max_val)
        else:
            max_indices.append(None)

    highlight_methods = set(highlight_methods or [])

    # Print table
    current_group = None
    for method_idx in range(row_count):
        method_name = method_names[method_idx]
        group, display_name = method_groups.get(method_name, ("Unknown", method_name))

        # Print group header
        if group != current_group:
            if current_group is not None:
                print("\\midrule")

            # Count rows in this group
            group_count = sum(1 for m in method_names[method_idx:] if method_groups.get(m, ("", ""))[0] == group)
            print(f"\\multirow{{{group_count}}}{{*}}{{{group}}}")
            current_group = group

        row_data = all_data[method_idx]

        # Highlight?
        is_highlight = method_name in highlight_methods

        # Format values
        formatted_values = []
        for col_idx, val in enumerate(row_data):
            formatted_val = fmt(val)

            # Bold max values
            if val is not None and max_indices[col_idx] is not None and abs(val - max_indices[col_idx]) < 1e-6:
                formatted_val = f"\\textbf{{{formatted_val}}}"

            # Apply highlight
            if is_highlight and formatted_val != "-":
                formatted_val = f"\\cellcolor{{Highlight}}{formatted_val}"

            formatted_values.append(formatted_val)

        # Print row
        if is_highlight:
            row = f"& \\cellcolor{{Highlight}}{display_name} \n& " + " & ".join(formatted_values) + " \\\\"
        else:
            row = f"& {display_name} & " + " & ".join(formatted_values) + " \\\\"

        print(row)

    if print_footer:
        print("\\hline")


def print_combined_lpm_mpm_hpm_table(all_path_results, method_names):
    highlight_methods = {"probe", "dynamic"}
    print("\\renewcommand{\\arraystretch}{1.2}")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Scenario alignment ability of routing strategies across multiple benchmarks.}")
    print("\\small")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{llccccccccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Type}}")
    print("& \\multirow{2}{*}{\\textbf{Method}}")
    print("& \\multicolumn{4}{c}{\\textbf{In Domain}}")
    print("& \\multicolumn{7}{c}{\\textbf{Out of Domain}} \\\\")
    print("\\cmidrule(lr){3-6} \\cmidrule(lr){7-13}")
    print("& & Alpaca & Big Math & MMLU & AVG")
    print("& Magpie & MATH & STEM & Humanities & Social Sciences & Others & AVG \\\\")
    print("\\midrule")

    section_specs = [
        ("LPM", "LPM (Low Performance Mean)"),
        ("MPM", "MPM (Middle Performance Mean)"),
        ("HPM", "HCR(High-band Call Rate)"),
    ]
    for metric, title in section_specs:
        print(f"\\multicolumn{{13}}{{c}}{{\\textit{{{title}}}}} \\\\")
        print("\\midrule")
        print_metric_comparison_table(
            all_path_results,
            metric,
            method_names,
            highlight_methods=highlight_methods,
            print_header=False,
            print_footer=False,
        )
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\label{tab:scence}")
    print("\\end{table*}")


def print_auroc_table(all_path_results, method_names):
    highlight_methods = {"probe", "dynamic"}
    print("\\renewcommand{\\arraystretch}{1.2}")
    print("\\begin{table*}[!t]")
    print("\\centering")
    print("\\caption{Router ability (AUROC) comparison of routing strategies across multiple benchmarks.}")
    print("\\small")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{l|l|cccc|ccccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Type}}")
    print("& \\multirow{2}{*}{\\textbf{Method}}")
    print("& \\multicolumn{4}{c|}{\\textbf{In Domain}}")
    print("& \\multicolumn{7}{c}{\\textbf{Out of Domain}} \\\\")
    print("\\cmidrule(lr){3-6} \\cmidrule(lr){7-13}")
    print("& & Alpaca & Big Math & MMLU & \\textbf{AVG}")
    print("& Magpie & MATH & STEM & Humanities & Social Sciences & Others & \\textbf{AVG} \\\\")
    print("\\midrule")
    print_metric_comparison_table(
        all_path_results,
        "auroc",
        method_names,
        highlight_methods=highlight_methods,
        print_header=False,
        print_footer=False,
    )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\label{tab:perf_char}")
    print("\\end{table*}")

def scan_methods(base_path):
    """Scan all method folders under base_path.

    Args:
        base_path: base path, e.g. .../metric_results/base

    Returns:
        list: [(method_name, method_dir_path), ...]
    """
    methods = []

    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return methods

    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        # Only directories; skip files (e.g., summary.json)
        if os.path.isdir(item_path):
            methods.append((item, item_path))

    # Sort by method name
    methods.sort(key=lambda x: x[0])

    return methods


def load_probe_results_from_train_set(train_set_name):
    """Load probe results from a training-set directory.

    Args:
        train_set_name: training set name, e.g. mmlu, alpaca_5k_train, big_math_5k

    Returns:
        (individual_results, mmlu_pro_averages): same result format as other loaders
    """
    metric_results_base = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results"
    train_dir = os.path.join(metric_results_base, train_set_name)

    if not os.path.exists(train_dir):
        print(f"Error: Train directory does not exist: {train_dir}")
        return {}, {}

    # Use "probe" method name when loading data
    print(f"Loading probe results from: {train_dir}")
    individual_results = load_individual_datasets(train_dir, "probe")
    mmlu_pro_averages = load_mmlu_pro_categories(train_dir, "probe")

    return individual_results, mmlu_pro_averages


def main():
    import sys

    # Parse CLI args
    mode = "base"  # default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    metric_results_base = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results"

    print(f"Running in mode: {mode}")
    print("="*80)

    all_path_results = []
    method_names = []

    if mode == "base":
        # Original base mode
        base_path = os.path.join(metric_results_base, "base")

        print(f"Scanning methods in: {base_path}")
        methods = scan_methods(base_path)

        if not methods:
            print("No methods found!")
            return

        print(f"Found {len(methods)} methods:")
        for method_name, method_path in methods:
            print(f"  - {method_name}")

        results_by_method = {}
        for method_name, method_dir in methods:
            print(f"\n{'='*80}")
            print(f"Loading data for method: {method_name}")
            print(f"{'='*80}")

            # Special-case probe_sampling directory (subdirs end with _dynamic_fusion_sampling)
            if method_name == "probe_sampling":
                actual_method_name = "dynamic_fusion"
            elif method_name == "deberta":
                actual_method_name = "trained_deberta"
            else:
                actual_method_name = method_name

            print("Loading individual dataset metrics...")
            individual_results = load_individual_datasets(method_dir, actual_method_name)

            print("Loading MMLU Pro category metrics...")
            mmlu_pro_averages = load_mmlu_pro_categories(method_dir, actual_method_name)

            results_by_method[actual_method_name] = (individual_results, mmlu_pro_averages)

            print(f"Loaded {sum(1 for r in individual_results.values() if r is not None)} individual datasets")
            print(f"Loaded {sum(1 for r in mmlu_pro_averages.values() if r is not None)} MMLU Pro categories")

    elif mode in ["mmlu", "alpaca", "big_math","alpaca+big_math","dynamic","test"]:
        # Probe mode: load probe results from training-set directory
        train_set_mapping = {
            "mmlu": "mmlu",
            "alpaca": "alpaca_5k_train",
            "big_math": "big_math_5k",
            "alpaca+big_math":"alpaca+big_math",
            "dynamic":"dynamic",
            "test":"test"
        }

        train_set_name = train_set_mapping[mode]
        print(f"Loading probe results for training set: {train_set_name}")

        individual_results, mmlu_pro_averages = load_probe_results_from_train_set(train_set_name)

        all_path_results.append((individual_results, mmlu_pro_averages))
        method_names.append(f"probe_{train_set_name}")

        print(f"Loaded {sum(1 for r in individual_results.values() if r is not None)} individual datasets")
        print(f"Loaded {sum(1 for r in mmlu_pro_averages.values() if r is not None)} MMLU Pro categories")

    else:
        print(f"Error: Unknown mode '{mode}'")
        print("Available modes: base, mmlu, alpaca, big_math")
        return

    # Print comparison tables
    if mode == "base":
        desired_methods = [
            "self_questioning",
            "semantic_entropy",
            "confidence_margin",
            "entropy",
            "max_logits",
            "embedding_mlp",
            "trained_deberta",
            "probe",
            "dynamic",
        ]

        target_datasets = ["alpaca_5k_test", "big_math_5k_test", "mmlu_test", "magpie_5k_test", "math"]
        mmlu_pro_order = ["STEM", "Humanities", "Social Science", "Other"]

        all_path_results = []
        method_names = []
        for method in desired_methods:
            if method in results_by_method:
                all_path_results.append(results_by_method[method])
                method_names.append(method)
            else:
                individual_results = {k: None for k in target_datasets}
                mmlu_pro_averages = {k: None for k in mmlu_pro_order}
                all_path_results.append((individual_results, mmlu_pro_averages))
                method_names.append(method)

        print_combined_lpm_mpm_hpm_table(all_path_results, method_names)
        print_auroc_table(all_path_results, method_names)
    else:
        metrics = ["auroc", "LPM", "HPM", "MPM"]

        for metric in metrics:
            print(f"\n{'='*100}")
            print(f"Processing {metric.upper()}")
            print(f"{'='*100}")

            print_metric_comparison_table(all_path_results, metric, method_names)
            print("\n" + "="*100)

if __name__ == "__main__":
    main()
