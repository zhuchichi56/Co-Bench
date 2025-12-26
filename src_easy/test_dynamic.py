#!/usr/bin/env python3
"""
最简动态融合 probe 测试脚本
- CLI 参数保持不变
- 只保留：读取数据 ->（可选采样）-> 训练/测试 ->（Dirichlet 可选）不确定性评估
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

from dynamic_probe import run_dynamic_probe_pipeline, DynamicFusionProbe, DynamicProbeDataset


# ----------------------------
# Utilities
# ----------------------------
def load_and_merge_data(test_configs: List[Dict[str, str]], max_samples: Optional[int] = None) -> List[Any]:
    """加载多个 .pt 数据并合并；如 max_samples 指定，则随机采样到该数量。"""
    all_data = []

    for cfg in test_configs:
        task = cfg["task"]
        path = cfg["hidden_states_file"]

        if not os.path.exists(path):
            print(f"[WARN] File not found, skip: {path}")
            continue

        print(f"Loading: {task} from {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        all_data.extend(data)
        print(f"  Loaded {len(data)} samples")

    if not all_data:
        raise RuntimeError("No valid data loaded from given configs.")

    if max_samples is not None and len(all_data) > max_samples:
        # 这里用 torch.randperm，避免引入 random
        idx = torch.randperm(len(all_data))[:max_samples].tolist()
        all_data = [all_data[i] for i in idx]
        print(f"Sampled to max_samples={max_samples}. Final size: {len(all_data)}")
    else:
        print(f"Final merged size: {len(all_data)}")

    return all_data


def evaluate_uncertainty_dirichlet(model_path: str, eval_data: List[Any], num_samples: int = 50) -> Optional[Dict[str, float]]:
    """仅对 Dirichlet probe 做不确定性评估（MC 采样）。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    meta = ckpt.get("metadata", {})
    probe_type = meta.get("probe_type", "softmax")

    if probe_type != "dirichlet":
        print("Uncertainty evaluation only supports Dirichlet probes. Skip.")
        return None

    model = DynamicFusionProbe(
        meta["input_dim"],
        meta["num_layers"],
        meta["output_dim"],
        probe_type=probe_type,
        mlp_hidden_dims=meta.get("mlp_hidden_dims", None),
        dropout=meta.get("dropout", 0.1),
        use_input_dependent=meta.get("use_input_dependent", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(DynamicProbeDataset(eval_data), batch_size=32, shuffle=False)

    all_unc = []
    all_pred = []
    all_y = []

    print(f"Evaluating Dirichlet uncertainty with num_samples={num_samples}")

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # MC sampling: 每次 forward 时强制 train() 以启用采样/随机性（按你原逻辑）
            preds = []
            uncs = []
            for _ in range(num_samples):
                model.train()
                logits, unc = model(x, return_uncertainty=True)
                p = torch.sigmoid(logits).squeeze(-1)

                preds.append(p.detach().cpu().numpy())
                uncs.append(unc.detach().cpu().numpy())

            preds = np.asarray(preds)            # [S, B]
            uncs = np.asarray(uncs)              # [S, B]
            pred_mean = preds.mean(axis=0)       # [B]
            unc_mean = uncs.mean(axis=0)         # [B]

            all_pred.append(pred_mean)
            all_unc.append(unc_mean)
            all_y.append(y.numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_unc = np.concatenate(all_unc, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    bin_pred = (all_pred > 0.5).astype(int)
    correct = (bin_pred == all_y)

    correct_unc = all_unc[correct]
    wrong_unc = all_unc[~correct]

    # 注意：全对或全错时 corrcoef 会产生 NaN，这里做个保护
    corr = np.corrcoef(all_unc, correct.astype(float))[0, 1]
    if np.isnan(corr):
        corr = 0.0

    stats = {
        "mean_uncertainty": float(all_unc.mean()),
        "std_uncertainty": float(all_unc.std()),
        "correct_mean_uncertainty": float(correct_unc.mean()) if correct_unc.size else 0.0,
        "incorrect_mean_uncertainty": float(wrong_unc.mean()) if wrong_unc.size else 0.0,
        "uncertainty_accuracy_correlation": float(corr),
    }

    print("Uncertainty stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return stats


def train_and_test(
    task_name: str,
    data_file: str,
    probe_type: str,
    mlp_hidden_dims: Optional[List[int]],
    dropout: float,
    save_dir: str,
    use_input_dependent: bool,
    epochs: int,
) -> Dict[str, Any]:
    """单次训练/测试封装。"""
    results = run_dynamic_probe_pipeline(
        task=task_name,
        hidden_states_file=data_file,
        save_dir=save_dir,
        probe_type=probe_type,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        use_input_dependent=use_input_dependent,
        epochs=epochs,
    )

    tr = results["training_results"]
    te = results["test_results"]

    print("\nTraining results:")
    print(f"  best_val_loss: {tr['best_val_loss']:.4f}")
    print(f"  final_layer_weights: {tr['final_layer_weights']}")

    print("\nTest results:")
    print(f"  accuracy: {te['accuracy']:.4f}")

    if probe_type == "dirichlet":
        print(f"  global_concentration(beta0): {te.get('global_concentration', 0.0):.4f}")

    print(f"\nModel saved to: {results['model_path']}")
    return results


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Fusion Probe Training and Testing (minimal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_dynamic.py --datasets alpaca_5k mmlu_train --probe_types softmax
  python test_dynamic.py --datasets alpaca_5k mmlu_train big_math --probe_types softmax dirichlet --max_samples 12000 --mlp_hidden_dims 64 128 --dropout 0.5 --save_dir custom/save/path
  python test_dynamic.py --datasets alpaca_5k --probe_types dirichlet --max_samples 5000
        """,
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["alpaca_5k", "mmlu_train", "big_math"],
        help="数据集名称列表（空格分隔），可选: alpaca_5k, mmlu_train, big_math",
    )
    parser.add_argument(
        "--probe_types",
        type=str,
        nargs="+",
        default=["softmax", "dirichlet"],
        help="Probe 类型列表（空格分隔），可选: softmax, dirichlet",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=12000,
        help="最大采样数量（默认: 12000）",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        type=int,
        nargs="*",
        default=None,
        help="MLP 隐藏层维度列表（空格分隔），例如: 64 128。留空表示单层线性分类器",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout 比率（默认: 0.1）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/",
        help="模型和历史保存目录",
    )
    parser.add_argument(
        "--use_input_dependent",
        action="store_true",
        help="是否使用输入依赖的 Dirichlet（仅对 dirichlet 类型有效）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数（默认: 50）",
    )

    args = parser.parse_args()

    print("Running with parameters:")
    print(f"  datasets: {args.datasets}")
    print(f"  probe_types: {args.probe_types}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  mlp_hidden_dims: {args.mlp_hidden_dims}")
    print(f"  dropout: {args.dropout}")
    print(f"  save_dir: {args.save_dir}")
    print(f"  use_input_dependent: {args.use_input_dependent}")
    print(f"  epochs: {args.epochs}")
    print("")

    dataset_map = {
        "alpaca_5k": {
            "task": "alpaca_5k_train",
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Qwen3-8B_alpaca_5k_train.pt",
        },
        "mmlu_train": {
            "task": "mmlu_train",
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Qwen3-8B_mmlu_train.pt",
        },
        "big_math": {
            "task": "big_math_5k_train",
            "hidden_states_file": "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/hs/Qwen3-8B_big_math_5k_train.pt",
        },
    }

    # 构建 test_configs
    test_configs = []
    for name in args.datasets:
        if name not in dataset_map:
            print(f"[WARN] Unknown dataset '{name}', skip.")
            continue
        test_configs.append(dataset_map[name])

    if not test_configs:
        raise RuntimeError("No valid datasets specified after filtering.")

    # 读取并合并（可采样）
    merged = load_and_merge_data(test_configs, max_samples=args.max_samples)

    # 写一个临时文件给 pipeline 读（保持你原 run_dynamic_probe_pipeline 的接口）
    tmp_dir = os.path.join(os.path.dirname(__file__), "..")
    tmp_file = os.path.abspath(os.path.join(tmp_dir, f"temp_mixed_data.pt"))
    os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
    torch.save(merged, tmp_file)

    task_name = "mixed_" + "_".join([cfg["task"] for cfg in test_configs])

    all_results = {}

    try:
        for probe_type in args.probe_types:
            print("\n" + "=" * 80)
            print(f"Training probe_type={probe_type} on task={task_name}")
            print("=" * 80)

            results = train_and_test(
                task_name=task_name,
                data_file=tmp_file,
                probe_type=probe_type,
                mlp_hidden_dims=args.mlp_hidden_dims,
                dropout=args.dropout,
                save_dir=args.save_dir,
                use_input_dependent=args.use_input_dependent,
                epochs=args.epochs,
            )

            summary = {
                "accuracy": results["test_results"]["accuracy"],
                "best_val_loss": results["training_results"]["best_val_loss"],
                "total_samples": len(merged),
                "layer_weights": results["test_results"]["layer_weights"].tolist(),
                "model_path": results["model_path"],
            }

            # Dirichlet 不确定性评估：用同样的“后 20%”作为 eval（保持你原先的习惯）
            if probe_type == "dirichlet":
                split = int(len(merged) * 0.8)
                eval_data = merged[split:]
                stats = evaluate_uncertainty_dirichlet(results["model_path"], eval_data, num_samples=50)
                if stats is not None:
                    summary["uncertainty_stats"] = stats
                    summary["global_concentration"] = results["test_results"].get("global_concentration", None)
                    summary["concentration"] = results["test_results"].get("concentration", None)

            all_results[probe_type] = summary

        # 打印汇总表
        print("\nSummary:")
        print(f"{'Probe':<10} {'Accuracy':<10} {'ValLoss':<10} {'Samples':<10}")
        print("-" * 50)
        for p, s in all_results.items():
            print(f"{p:<10} {s['accuracy']:<10.4f} {s['best_val_loss']:<10.4f} {s['total_samples']:<10d}")

        # 可选：落盘结果（保留一个最基础的 json）
        out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "dynamic_probe_minimal_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to: {out_path}")

    finally:
        # 清理临时文件
        try:
            os.remove(tmp_file)
        except Exception:
            pass


if __name__ == "__main__":
    main()
