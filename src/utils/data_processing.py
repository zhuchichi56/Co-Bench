import json
import os
os.environ["WANDB_API_KEY"] = "79a88980fe13540412ac35e9673ca1ebe5e23380"
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def extract_method_from_dirname(dirname):
    """从目录名中提取方法名
    例如: alpaca_5k_test_coe -> coe
         mmlu_pro_biology_entropy -> entropy
    """
    known_methods = [
        "coe", "entropy", "confidence_margin", "max_logits",
        "top10_variance", "semantic_entropy", "self_questioning",
        "dynamic_dirichlet", "probe"
    ]

    for method in known_methods:
        if dirname.endswith(f"_{method}"):
            return method

    return None

def extract_dataset_from_dirname(dirname, method):
    """从目录名中提取数据集名
    例如: alpaca_5k_test_coe (method=coe) -> alpaca_5k_test
         mmlu_pro_biology_entropy (method=entropy) -> mmlu_pro_biology
    """
    if method and dirname.endswith(f"_{method}"):
        return dirname[:-len(f"_{method}")]
    return dirname

def load_individual_datasets(method_dir, method_name):
    """读取单独数据集的metrics.json文件

    Args:
        method_dir: 方法目录路径，例如 .../base/coe
        method_name: 方法名，例如 coe
    """

    # 标准数据集名（不包含_test后缀的也要匹配）
    target_datasets = ["alpaca_5k_test", "big_math_5k_test", "mmlu_test", "magpie_5k_test", "math"]

    results = {}

    # 遍历方法目录下的所有文件夹
    for item in os.listdir(method_dir):
        item_path = os.path.join(method_dir, item)
        if not os.path.isdir(item_path):
            continue

        # 提取数据集名
        dataset_name = extract_dataset_from_dirname(item, method_name)

        # 检查是否是目标数据集（不是mmlu_pro）
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

                results[dataset_name] = metrics

            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
                results[dataset_name] = None
        else:
            print(f"Metrics file not found: {metrics_file}")
            results[dataset_name] = None

    # 补充未找到的数据集为None
    for dataset in target_datasets:
        if dataset not in results:
            results[dataset] = None

    return results

def load_mmlu_pro_categories(method_dir, method_name):
    """读取MMLU Pro分类数据

    Args:
        method_dir: 方法目录路径，例如 .../base/coe
        method_name: 方法名，例如 coe
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

        # 提取数据集名
        dataset_name = extract_dataset_from_dirname(item, method_name)

        # 只处理mmlu_pro开头的
        if not dataset_name.startswith("mmlu_pro_"):
            continue

        # 提取subject名字
        subject = dataset_name.replace("mmlu_pro_", "")

        # 找到对应的category
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

    # 计算每个category的平均值
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


def print_metric_comparison_table(all_path_results, metric, method_names):
    """为特定指标打印对比表格（带格式、加粗最大值、高亮最后一行）"""

    dataset_order = ["alpaca_5k_test", "big_math_5k_test", "mmlu_test", "magpie_5k_test", "math"]
    dataset_display_names = ["Alpaca", "Numina", "MMLU", "Magpie", "MATH"]

    first_three_keys = dataset_order[:3]
    first_three_names = dataset_display_names[:3]
    last_two_keys = dataset_order[3:]
    last_two_names = dataset_display_names[3:]

    mmlu_pro_order = ["STEM", "Humanities", "Social Science", "Other"]

    # 方法分组映射
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
        "probe": ("Probe", "Probe"),
        # "dynamic": ("Probe", "DynamicDirichlet"),
        # "dynamic_dirichlet": ("Probe", "DynamicDirichlet")
    }

    def fmt(x):
        """将[0,1]的小数转成百分数并保留两位小数；None -> '-'"""
        if x is None:
            return "-"
        return f"{x * 100:.2f}"

    print(f"\n{metric.upper()} Comparison Table:")
    print("=" * 200)

    all_columns = first_three_names + ["Avg(1-3)"] + last_two_names + mmlu_pro_order + ["Avg(4-9)"]
    header = "Method & " + " & ".join([f"\\textbf{{{col}}}" for col in all_columns]) + " \\\\"
    print(header)
    print("\\hline")

    # 收集所有数据用于找最大值
    all_data = []
    row_count = min(len(method_names), len(all_path_results))
    
    for method_idx in range(row_count):
        method_name = method_names[method_idx]
        individual_results, mmlu_pro_averages = all_path_results[method_idx]

        row_data = []

        # 前三个数据集
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

        # 后两个数据集
        last_two_vals = []
        for ds in last_two_keys:
            v = None if individual_results.get(ds) is None else individual_results[ds].get(metric, None)
            row_data.append(v)
            if v is not None:
                last_two_vals.append(v)

        # 四个 MMLU-Pro 类别
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

    # 找每列的最大值
    num_cols = len(all_data[0])
    max_indices = []
    for col_idx in range(num_cols):
        col_values = [row[col_idx] for row in all_data if row[col_idx] is not None]
        if col_values:
            max_val = max(col_values)
            max_indices.append(max_val)
        else:
            max_indices.append(None)

    # 打印表格
    current_group = None
    for method_idx in range(row_count):
        method_name = method_names[method_idx]
        group, display_name = method_groups.get(method_name, ("Unknown", method_name))

        # 打印分组头
        if group != current_group:
            if current_group is not None:
                print("\\midrule")
            
            # 计算该组有多少行
            group_count = sum(1 for m in method_names[method_idx:] if method_groups.get(m, ("", ""))[0] == group)
            print(f"\\multirow{{{group_count}}}{{*}}{{{group}}}")
            current_group = group

        row_data = all_data[method_idx]
        
        # 是否是最后一行（dynamic）
        is_last_row = (method_name == "dynamic")
        
        # 格式化每个值
        formatted_values = []
        for col_idx, val in enumerate(row_data):
            formatted_val = fmt(val)
            
            # 如果是最大值，加粗
            if val is not None and max_indices[col_idx] is not None and abs(val - max_indices[col_idx]) < 1e-6:
                formatted_val = f"\\textbf{{{formatted_val}}}"
            
            # 如果是最后一行，加高亮
            if is_last_row and formatted_val != "-":
                formatted_val = f"\\cellcolor{{Highlight}}{formatted_val}"
            
            formatted_values.append(formatted_val)

        # 打印行
        if is_last_row:
            row = f"& \\cellcolor{{Highlight}}{display_name} \n& " + " & ".join(formatted_values) + " \\\\"
        else:
            row = f"& {display_name} & " + " & ".join(formatted_values) + " \\\\"
        
        print(row)

    print("\\hline")

def scan_methods(base_path):
    """自动扫描base_path下的所有方法文件夹

    Args:
        base_path: 基础路径，例如 .../metric_results/base

    Returns:
        list: [(method_name, method_dir_path), ...]
    """
    methods = []

    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return methods

    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        # 只处理目录，跳过文件（如summary.json）
        if os.path.isdir(item_path):
            methods.append((item, item_path))

    # 按方法名排序
    methods.sort(key=lambda x: x[0])

    return methods


def main():
    # 基础路径
    base_path = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/base"

    # 自动扫描所有方法
    print(f"Scanning methods in: {base_path}")
    methods = scan_methods(base_path)

    if not methods:
        print("No methods found!")
        return

    print(f"Found {len(methods)} methods:")
    for method_name, method_path in methods:
        print(f"  - {method_name}")

    all_path_results = []
    method_names = []

    for method_name, method_dir in methods:
        print(f"\n{'='*80}")
        print(f"Loading data for method: {method_name}")
        print(f"{'='*80}")

        print("Loading individual dataset metrics...")
        individual_results = load_individual_datasets(method_dir, method_name)

        print("Loading MMLU Pro category metrics...")
        mmlu_pro_averages = load_mmlu_pro_categories(method_dir, method_name)

        all_path_results.append((individual_results, mmlu_pro_averages))
        method_names.append(method_name)

        print(f"Loaded {sum(1 for r in individual_results.values() if r is not None)} individual datasets")
        print(f"Loaded {sum(1 for r in mmlu_pro_averages.values() if r is not None)} MMLU Pro categories")

    # 打印对比表格
    metrics = ["auroc", "LPM", "HPM", "MPM"]

    for metric in metrics:
        print(f"\n{'='*100}")
        print(f"Processing {metric.upper()}")
        print(f"{'='*100}")

        print_metric_comparison_table(all_path_results, metric, method_names)
        print("\n" + "="*100)

if __name__ == "__main__":
    main()