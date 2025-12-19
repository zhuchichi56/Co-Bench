from config import PipelineConfig
from pipeline import RouterEvaluationPipeline, get_task_score
from train_router import (
    generate_logits, set_random_seed,
    complete_probe_training_pipeline_with_mixed_datasets,
    train_probe_model
)
import os
config_env = PipelineConfig.from_yaml()
if config_env.inference.cuda_visible_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = config_env.inference.cuda_visible_devices
import copy
import json
import argparse
import glob
import re
from pathlib import Path
from datetime import datetime


def save_training_history(history, probe_type, task_list, max_samples=None, save_dir="probe_save/loss"):
    """
    Save training loss and accuracy history to a JSON file

    Args:
        history: Dictionary containing train_losses, val_losses, and best_val_loss
        probe_type: Type of probe being trained
        task_list: List of tasks used for training
        max_samples: Maximum number of samples used for training
        save_dir: Directory to save the history file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks_str = "_".join(task_list)
    filename = f"{tasks_str}_{probe_type}_{timestamp}.json"
    filepath = save_path / filename

    # Extract training results from history dict
    # The history dict has a 'training_results' key that contains the actual loss data
    training_results = history.get('training_results', {})
    train_losses = training_results.get('train_losses', [])
    val_losses = training_results.get('val_losses', [])
    val_accuracies = training_results.get('val_accuracies', [])
    val_aurocs = training_results.get('val_aurocs', [])
    learning_rates = training_results.get('learning_rates', [])
    best_val_loss = training_results.get('best_val_loss', float('inf'))
    best_val_acc = training_results.get('best_val_acc', 0.0)
    best_val_auroc = training_results.get('best_val_auroc', 0.0)
    initial_lr = training_results.get('initial_lr', 0.0)

    # Save to JSON
    save_data = {
        "probe_type": probe_type,
        "tasks": task_list,
        "datasets": task_list,  # 添加数据集信息(与tasks相同)
        "max_samples": max_samples,  # 添加使用的最大样本数
        "timestamp": timestamp,
        "initial_lr": initial_lr,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_auroc": best_val_auroc,
        "epochs": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_aurocs": val_aurocs,
        "learning_rates": learning_rates,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"📊 Training history saved to: {filepath}")
    print(f"   Datasets used: {', '.join(task_list)}")
    if max_samples:
        print(f"   Max samples: {max_samples} ({max_samples/1000:.1f}k)")
    return str(filepath)


def batch_evaluate_probes(base_config, probe_configs, eval_tasks):
    for i, probe_config in enumerate(probe_configs):
        config_copy = copy.deepcopy(base_config)
        config_copy.router.checkpoint_path = probe_config['checkpoint_path']
        config_copy.router.probe_type = probe_config['probe_type']
        config_copy.metric_results_dir = probe_config['metric_results_dir']
        print(f"\n{'='*60}")
        print(f"Running evaluation {i+1}/{len(probe_configs)}")
        print(f"Probe type: {probe_config['probe_type']}")
        print(f"Checkpoint: {probe_config['checkpoint_path']}")
        print(f"{'='*60}")
        pipeline = RouterEvaluationPipeline(config_copy)

        for task in eval_tasks:
            print(f"\nEvaluating task: {task}")

            hidden_states_file = _build_hs_path(task)

            if not os.path.exists(hidden_states_file):
                print(f"Warning: Hidden states file not found: {hidden_states_file}")
                continue

            datasets = [f"{task}"]

            pipeline.evaluate_complete_pipeline(
                hidden_states_file=hidden_states_file,
                datasets=datasets
            )


def _build_hs_path(task: str):
    """构建hidden states文件路径"""
    base_dir = os.path.join("..", "hs")
    if task.startswith("mmlu_pro_"):
        base_dir = os.path.join(base_dir, "mmlu_pro")
    return os.path.join(base_dir, f"Llama-3.1-8B-Instruct_{task}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CoBench Router Evaluation and Training Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mode", type=str, required=True,
                       choices=["get_scores", "get_logits", "train", "eval_probe",
                               "eval_max_k", 
                               "self_based", "logits_based_routers"],
                       help="运行模式")

    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                       help="要处理的数据集列表 (空格分隔)")

    parser.add_argument("--probe_types", type=str, nargs="+",
                       default=["hs_last_mlp", "mean", "max", "coe_dual_mlp"],
                       help="训练模式下的 probe 类型")

    parser.add_argument("--max_samples", type=int, default=4000,
                       help="训练时的最大样本数")

    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    parser.add_argument("--save_loss_history", action="store_true",
                       help="在训练时保存 loss 和 accuracy 历史记录")

    parser.add_argument("--probe_dir", type=str, default=None,
                       help="Probe 模型目录路径（用于 eval_probe 模式）")


    args = parser.parse_args()
    mode = args.mode
    set_random_seed(args.seed)
    config = PipelineConfig().from_yaml()
    pipeline = RouterEvaluationPipeline(config)

    # ==================== 模式: get_scores ====================
    if mode == "get_scores":
        datasets = args.datasets 
        print(f"🎯 获取以下数据集的 scores: {datasets}")
        for task in datasets:
            print(f"\n{'='*60}")
            print(f"📊 处理任务: {task}")
            print(f"{'='*60}")
            try:
                score = get_task_score(config, task=task)
                print(f"✅ {task} 完成")
                if score:
                    print(f"   Score path: {score}")
            except Exception as e:
                print(f"❌ {task} 失败: {e}")

    # ==================== 模式: get_logits ====================
    elif mode == "get_logits":
        datasets = args.datasets 
        print(f"🎯 生成以下数据集的 logits: {datasets}")

        for task in datasets:
            print(f"\n{'='*60}")
            print(f"📊 处理任务: {task}")
            print(f"{'='*60}")

            # 构建 task_path
            if task.startswith("mmlu_pro_"):
                task_path = os.path.join("./results/mmlu_pro", f"{task}.jsonl")
            else:
                task_path = os.path.join("./results", f"{task}.jsonl")

            if not os.path.exists(task_path):
                print(f"⚠️  警告: 结果文件 {task_path} 不存在")
                print(f"   请先运行: python run.py --mode get_scores --datasets {task}")
                continue

            try:
                generate_logits(config, task=task, task_path=task_path)
                print(f"✅ {task} logits 生成完成")
            except Exception as e:
                print(f"❌ {task} logits 生成失败: {e}")

    # ==================== 模式: train ====================
    elif mode == "train":
        datasets = args.datasets 
        probe_types = args.probe_types
        max_samples = args.max_samples
        save_history = args.save_loss_history

        print(f"🚀 开始训练 Probe 模型")
        print(f"📊 数据集: {datasets}")
        print(f"🔧 Probe 类型: {probe_types}")
        print(f"📈 最大样本数: {max_samples}")
        print(f"💾 保存训练历史: {save_history}")

        for probe_type in probe_types:
            print(f"\n{'='*60}")
            print(f"🎯 训练 Probe 类型: {probe_type}")
            print(f"{'='*60}")

            config.router.probe_type = probe_type

            try:
                # 训练并获取历史记录
                history = complete_probe_training_pipeline_with_mixed_datasets(
                    config,
                    task_list=datasets,
                    max_samples=max_samples,
                    mix_strategy="balanced"
                )

                print(f"✅ {probe_type} 训练完成")

                # 保存训练历史
                if save_history and history:
                    save_training_history(history, probe_type, datasets, max_samples)

            except Exception as e:
                print(f"❌ {probe_type} 训练失败: {e}")
                import traceback
                traceback.print_exc()

    # ==================== 模式: eval_probe ====================
    elif mode == "eval_probe":
        datasets = args.datasets 
        probe_types = args.probe_types
        probe_dir = args.probe_dir
        
        print(f"🎯 评估 Probe 模型")
        print(f"📊 数据集: {datasets}")
        print(f"🔧 Probe 类型: {probe_types}")
        if probe_dir:
            print(f"📁 Probe 目录: {probe_dir}")

        # 如果指定了 probe_dir，从目录中加载所有 probe 文件
        if probe_dir:
            probe_files = sorted(glob.glob(f"{probe_dir}/*.pt"))
            print(f"\n在 {probe_dir} 中找到 {len(probe_files)} 个 probe 文件")

            probe_configs = []
            for pf in probe_files:
                # 尝试从文件名提取 probe_type
                filename = os.path.basename(pf)
                # 匹配模式: *_probe_type.pt 或 *_train_probe_type.pt
                m = re.search(r'.*?_(?:train_)?([^_]+)\.pt$', filename)
                if m:
                    detected_probe_type = m.group(1)
                    # 如果指定了 probe_types，只处理匹配的类型
                    if probe_types and detected_probe_type not in probe_types:
                        continue

                    metric_results_dir = config.metric_results_dir

                    probe_configs.append({
                        "checkpoint_path": pf,
                        "probe_type": detected_probe_type,
                        "metric_results_dir": metric_results_dir,
                    
                    })

            if probe_configs:
                batch_evaluate_probes(config, probe_configs, datasets)
            else:
                print(f"⚠️  警告: 在 {probe_dir} 中没有找到匹配的 probe 文件")

        # 如果没有指定 probe_dir，使用当前配置的 probe
        else:
            for probe_type in probe_types:
                print(f"\n{'='*60}")
                print(f" 使用 Probe 类型: {probe_type}")
                print(f"{'='*60}")

                config_copy = copy.deepcopy(config)
                config_copy.router.probe_type = probe_type
                # 如果配置中已有 checkpoint_path，使用它；否则需要用户指定
                if not config_copy.router.checkpoint_path:
                    print(f"⚠️  警告: 未指定 checkpoint_path，请在配置文件中设置或使用 --probe_dir")
                    continue

                pipeline_test = RouterEvaluationPipeline(config_copy)

                for task in datasets:
                    print(f"\n📊 评估任务: {task}")

                    hidden_states_file = _build_hs_path(task)

                    if not os.path.exists(hidden_states_file):
                        print(f"⚠️  警告: Hidden states 文件不存在: {hidden_states_file}")
                        continue

                    try:
                        results = pipeline_test.evaluate_complete_pipeline(
                            hidden_states_file=hidden_states_file,
                            datasets=[task]
                        )
                        print(f"✅ {task} 使用 {probe_type} 评估完成")
                    except Exception as e:
                        print(f"❌ {task} 使用 {probe_type} 评估失败: {e}")

    # ==================== 模式: eval_max_k ====================
    elif mode == "eval_max_k":
        probe_dir = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/max_k"
        probe_files = sorted(glob.glob(f"{probe_dir}/*.pt"))

        print(f"在 {probe_dir} 中找到 {len(probe_files)} 个 probe 文件")

        probe_configs = []
        for pf in probe_files:
            filename = os.path.basename(pf)
            sample_match = re.search(r'_cot_5k_train_(\w+)_(\d+k)\.pt$', filename)

            if sample_match:
                base_probe = sample_match.group(1)
                samples = sample_match.group(2)
                probe_type = base_probe

                metric_results_dir = f"metric_results/max_k/{samples}_{probe_type}"

                probe_configs.append({
                    "checkpoint_path": pf,
                    "probe_type": probe_type,
                    "metric_results_dir": metric_results_dir,
                })

        eval_datasets = args.datasets
        batch_evaluate_probes(config, probe_configs, eval_datasets)


    # ==================== 模式: self_based ====================
    elif mode == "self_based":
        strategies = [
            {
                "name": "semantic_entropy",
                "metric_results_dir": "metric_results/base/semantic_entropy",
                "num_samples": 5,
            },
            {
                "name": "self_questioning",
                "metric_results_dir": "metric_results/base/self_questioning",
                "num_samples": 8,
            },
        ]

        eval_datasets = args.datasets 

        for strat in strategies:
            print(f"\n{'='*60}")
            print(f"🚀 运行 self-based 策略: {strat['name']}")
            print(f"{'='*60}")

            config_copy = copy.deepcopy(config)
            config_copy.metric_results_dir = strat["metric_results_dir"]
            config_copy.router.router_type = strat["name"]
            config_copy.router.model_path = None
            config_copy.router.num_samples = strat["num_samples"]

            pipeline_test = RouterEvaluationPipeline(config_copy)

            for task in eval_datasets:
                print(f"\n评估任务: {task}")
                hidden_states_file = _build_hs_path(task)

                if not os.path.exists(hidden_states_file):
                    print(f"警告: Hidden states 文件不存在: {hidden_states_file}")
                    continue

                datasets = [task]
                pipeline_test.evaluate_complete_pipeline(
                    hidden_states_file=hidden_states_file,
                    datasets=datasets
                )

    # ==================== 模式: logits_based_routers ====================
    elif mode == "logits_based_routers":
        from router import RouterManager

        router_manager = RouterManager()
        router_manager.create_max_logits_router()
        router_manager.create_top10_variance_router()
        router_manager.create_coe_router()
        router_manager.create_entropy_router()
        router_manager.create_confidence_margin_router()

        router_types = ["max_logits", "top10_variance", "coe", "entropy", "confidence_margin"]
        eval_datasets = args.datasets 
        for router_type in router_types:
            print(f"\n{'='*60}")
            print(f"🚀 测试路由器: {router_type}")
            print(f"{'='*60}")

            for task in eval_datasets:
                print(f"\n评估任务: {task}")

                hidden_states_file = _build_hs_path(task)

                if not os.path.exists(hidden_states_file):
                    print(f"警告: Hidden states 文件不存在: {hidden_states_file}")
                    continue

                config_copy = copy.deepcopy(config)
                config_copy.router.router_type = router_type
                config_copy.metric_results_dir = f"metric_results/base/{router_type}"

                pipeline_test = RouterEvaluationPipeline(config_copy)

                datasets = [task]
                pipeline_test.evaluate_complete_pipeline(
                    hidden_states_file=hidden_states_file,
                    datasets=datasets
                )

    else:
        print(f"❌ 未知模式: {mode}")
        parser.print_help()

