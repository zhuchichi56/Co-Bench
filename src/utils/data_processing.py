import json
import os
os.environ["WANDB_API_KEY"] = "79a88980fe13540412ac35e9673ca1ebe5e23380"
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_individual_datasets(base_dir):
    """读取单独数据集的metrics.json文件"""
    
    datasets = ["alpaca_5k_test", "numina_cot_5k_test", "mmlu_test", "magpie_5k_test", "math"]
    
    results = {}
    
    for dataset in datasets:
        dataset_dir = None
        for item in os.listdir(base_dir):
            if item.startswith(f"{dataset}_"):
                dataset_dir = item
                break
        
        if dataset_dir is None and "semantic_entropy" in base_dir:
            for item in os.listdir(base_dir):
                if item == f"{dataset}_semantic_entropy":
                    dataset_dir = item
                    break
        
        if dataset_dir is None:
            print(f"Warning: Directory for {dataset} not found in {base_dir}")
            results[dataset] = None
            continue
        
        metrics_file = os.path.join(base_dir, dataset_dir, "metrics.json")
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
                
                if dataset in ["magpie_5k_test", "alpaca_5k_test"]:
                    metrics["LPM"] = metrics["LPM"] / 10
                    metrics["MPM"] = metrics["MPM"] / 10
                
                results[dataset] = metrics
                
            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
                results[dataset] = None
        else:
            print(f"Metrics file not found: {metrics_file}")
            results[dataset] = None
    
    return results

def load_mmlu_pro_categories(base_dir):
    """读取MMLU Pro分类数据"""
    
    categories = {
        "STEM": ["biology", "chemistry", "computer_science", "engineering", "math", "physics"],
        "Humanities": ["history", "philosophy"],  
        "Social Science": ["economics", "law", "psychology"],
        "Other": ["business", "health", "other"]
    }
    
    category_results = defaultdict(list)
    
    for item in os.listdir(base_dir):
        if item.startswith("mmlu_pro_"):
            temp = item.replace("mmlu_pro_", "")
            
            if temp.endswith("self_questioning"):
                subject = temp.rsplit("_", 2)[0]
            elif temp.endswith("semantic_entropy"):
                subject = temp.rsplit("_", 2)[0]
            elif temp.endswith("confidence_margin"):
                subject = temp.rsplit("_", 2)[0]
            elif temp.endswith("top10_variance"):
                subject = temp.rsplit("_", 2)[0]
            elif temp.endswith("max_logits"):
                subject = temp.rsplit("_", 2)[0]
            elif temp.endswith("dynamic_dirichlet"):
                subject = temp.rsplit("_", 2)[0] 
            else:
                known_suffixes = {"confidence", "max", "top10", "variance", "entropy", "coe"}
                parts = temp.split("_")
                if parts and parts[-1] in known_suffixes:
                    subject = "_".join(parts[:-1])
                else:
                    subject = temp.rsplit("_", 1)[0] if "_" in temp else temp
            
            category = None
            for cat, subjects in categories.items():
                if subject in subjects:
                    category = cat
                    break
            
            if category is None:
                print(f"Warning: Subject '{subject}' not found in any category")
                continue
            
            metrics_file = os.path.join(base_dir, item, "metrics.json")
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

# def load_individual_datasets(base_dir):
#     """读取单独数据集的metrics.json文件"""
    
#     datasets = ["alpaca_5k_test", "numina_cot_5k_test", "mmlu_test", "magpie_5k_test", "math"]
    
#     results = {}
    
#     for dataset in datasets:
#         dataset_dir = None
#         for item in os.listdir(base_dir):
#             if item.startswith(f"{dataset}_"):
#                 dataset_dir = item
#                 break
        
#         if dataset_dir is None and "semantic_entropy" in base_dir:
#             for item in os.listdir(base_dir):
#                 if item == f"{dataset}_semantic_entropy":
#                     dataset_dir = item
#                     break
        
#         if dataset_dir is None:
#             print(f"Warning: Directory for {dataset} not found in {base_dir}")
#             results[dataset] = None
#             continue
        
#         metrics_file = os.path.join(base_dir, dataset_dir, "metrics.json")
#         if os.path.exists(metrics_file):
#             try:
#                 with open(metrics_file, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
                
#                 # 读取 small_mean 和 large_mean
#                 small_mean = data["adaptive_metrics"]["small_mean"]
#                 large_mean = data["adaptive_metrics"]["large_mean"]
                
#                 # 读取原始 LPM 和 MPM
#                 lpm_raw = data["adaptive_metrics"]["LPM"]
#                 mpm_raw = data["adaptive_metrics"]["MPM"]
                
#                 # 计算相对提升
#                 if large_mean > small_mean:
#                     lpm_relative = (lpm_raw - small_mean) / (large_mean - small_mean)
#                     mpm_relative = (mpm_raw - small_mean) / (large_mean - small_mean)
#                 else:
#                     lpm_relative = 0.0
#                     mpm_relative = 0.0
                
#                 metrics = {
#                     "auroc": data["reliable_metrics"]["auroc"],
#                     "LPM": lpm_relative,  # 使用相对提升
#                     "HPM": data["adaptive_metrics"]["HPM"], 
#                     "MPM": mpm_relative   # 使用相对提升
#                 }
                
#                 # Magpie 和 Alpaca 不需要再除以10了，因为已经是相对提升
                
#                 results[dataset] = metrics
                
#             except Exception as e:
#                 print(f"Error reading {metrics_file}: {e}")
#                 results[dataset] = None
#         else:
#             print(f"Metrics file not found: {metrics_file}")
#             results[dataset] = None
    
#     return results

# def load_mmlu_pro_categories(base_dir):
#     """读取MMLU Pro分类数据"""
    
#     categories = {
#         "STEM": ["biology", "chemistry", "computer_science", "engineering", "math", "physics"],
#         "Humanities": ["history", "philosophy"],  
#         "Social Science": ["economics", "law", "psychology"],
#         "Other": ["business", "health", "other"]
#     }
    
#     category_results = defaultdict(list)
    
#     for item in os.listdir(base_dir):
#         if item.startswith("mmlu_pro_"):
#             temp = item.replace("mmlu_pro_", "")
            
#             if temp.endswith("self_questioning"):
#                 subject = temp.rsplit("_", 2)[0]
#             elif temp.endswith("semantic_entropy"):
#                 subject = temp.rsplit("_", 2)[0]
#             elif temp.endswith("confidence_margin"):
#                 subject = temp.rsplit("_", 2)[0]
#             elif temp.endswith("top10_variance"):
#                 subject = temp.rsplit("_", 2)[0]
#             elif temp.endswith("max_logits"):
#                 subject = temp.rsplit("_", 2)[0]
#             elif temp.endswith("dynamic_dirichlet"):
#                 subject = temp.rsplit("_", 2)[0] 
#             else:
#                 known_suffixes = {"confidence", "max", "top10", "variance", "entropy", "coe"}
#                 parts = temp.split("_")
#                 if parts and parts[-1] in known_suffixes:
#                     subject = "_".join(parts[:-1])
#                 else:
#                     subject = temp.rsplit("_", 1)[0] if "_" in temp else temp
            
#             category = None
#             for cat, subjects in categories.items():
#                 if subject in subjects:
#                     category = cat
#                     break
            
#             if category is None:
#                 print(f"Warning: Subject '{subject}' not found in any category")
#                 continue
            
#             metrics_file = os.path.join(base_dir, item, "metrics.json")
#             if os.path.exists(metrics_file):
#                 try:
#                     with open(metrics_file, 'r', encoding='utf-8') as f:
#                         data = json.load(f)
                    
#                     # 读取 small_mean 和 large_mean
#                     small_mean = data["adaptive_metrics"]["small_mean"]
#                     large_mean = data["adaptive_metrics"]["large_mean"]
                    
#                     # 读取原始 LPM 和 MPM
#                     lpm_raw = data["adaptive_metrics"]["LPM"]
#                     mpm_raw = data["adaptive_metrics"]["MPM"]
                    
#                     # 计算相对提升
#                     if large_mean > small_mean:
#                         lpm_relative = (lpm_raw - small_mean) / (large_mean - small_mean)
#                         mpm_relative = (mpm_raw - small_mean) / (large_mean - small_mean)
#                     else:
#                         lpm_relative = 0.0
#                         mpm_relative = 0.0
                    
#                     metrics = {
#                         "subject": subject,
#                         "auroc": data["reliable_metrics"]["auroc"],
#                         "LPM": lpm_relative,
#                         "HPM": data["adaptive_metrics"]["HPM"],
#                         "MPM": mpm_relative
#                     }
                    
#                     category_results[category].append(metrics)
                    
#                 except Exception as e:
#                     print(f"Error reading {metrics_file}: {e}")
    
#     averages = {}
#     category_order = ["STEM", "Humanities", "Social Science", "Other"]
    
#     for category in category_order:
#         if category in category_results and category_results[category]:
#             metrics_list = category_results[category]
#             averages[category] = {
#                 "auroc": sum(m["auroc"] for m in metrics_list) / len(metrics_list),
#                 "LPM": sum(m["LPM"] for m in metrics_list) / len(metrics_list),
#                 "HPM": sum(m["HPM"] for m in metrics_list) / len(metrics_list),
#                 "MPM": sum(m["MPM"] for m in metrics_list) / len(metrics_list)
#             }
#         else:
#             averages[category] = None
    
#     return averages

def print_metric_comparison_table(all_path_results, metric, method_names):
    """为特定指标打印对比表格（带格式、加粗最大值、高亮最后一行）"""

    dataset_order = ["alpaca_5k_test", "numina_cot_5k_test", "mmlu_test", "magpie_5k_test", "math"]
    dataset_display_names = ["Alpaca", "Numina", "MMLU", "Magpie", "MATH"]

    first_three_keys = dataset_order[:3]
    first_three_names = dataset_display_names[:3]
    last_two_keys = dataset_order[3:]
    last_two_names = dataset_display_names[3:]

    mmlu_pro_order = ["STEM", "Humanities", "Social Science", "Other"]

    # 方法分组映射
    method_groups = {
        "semantic_entropy": ("Generation", "SemanticEntropy"),
        "self_questioning": ("Generation", "SelfAsk"),
        "confidence_margin": ("Logit", "ConfidenceMargin"),
        "entropy": ("Logit", "Entropy"),
        "max_logits": ("Logit", "MaxLogits"),
        "coe_dual": ("Probe", "ProbeCoE"),
        "hs_last": ("Probe", "ProbeFinal"),
        "mean": ("Probe", "ProbeMean"),
        "dynamic": ("Probe", "DynamicDirichlet")
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

def main():
    
    base_directories = [
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/semantic_entropy",
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/self_questioning",
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/confidence_margin",
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/entropy",
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/max_logits",
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/coe_dual_mlp",
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/hs_last_mlp",
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/mean",
        "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/metric_results/weighted/dynamic_dirichlet"
    ]
    
    all_path_results = []
    
    for i, base_dir in enumerate(base_directories):
        print(f"\nLoading data from path {i+1}: {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"Warning: Directory not found: {base_dir}")
            all_path_results.append((
                {dataset: None for dataset in ["alpaca_5k_test", "numina_cot_5k_test", "mmlu_test", "magpie_5k_test", "math"]},
                {category: None for category in ["STEM", "Humanities", "Social Science", "Other"]}
            ))
            continue
        
        print("Loading individual dataset metrics...")
        individual_results = load_individual_datasets(base_dir)
        
        print("Loading MMLU Pro category metrics...")
        mmlu_pro_averages = load_mmlu_pro_categories(base_dir)
        
        all_path_results.append((individual_results, mmlu_pro_averages))
        
        print(f"Loaded {sum(1 for r in individual_results.values() if r is not None)} individual datasets")
        print(f"Loaded {sum(1 for r in mmlu_pro_averages.values() if r is not None)} MMLU Pro categories")
    
    metrics = ["auroc", "LPM", "HPM", "MPM"]
    method_names = [
        "semantic_entropy",
        "self_questioning",
        "confidence_margin",
        "entropy",
        "max_logits",
        "coe_dual",
        "hs_last",
        "mean",
        "dynamic"
    ]

    for metric in metrics:
        print(f"\n{'='*100}")
        print(f"Processing {metric.upper()}")
        print(f"{'='*100}")
        
        print_metric_comparison_table(all_path_results, metric, method_names)
        print("\n" + "="*100)

if __name__ == "__main__":
    main()