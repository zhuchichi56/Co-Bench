from config import PipelineConfig
from pipeline import RouterEvaluationPipeline, get_task_score
from train_router import (
    generate_logits, set_random_seed,
    complete_probe_training_pipeline_with_mixed_datasets,
    generate_query_embeddings,
    train_embedding_mlp_model,
    train_deberta_router,
    prepare_deberta_training_file
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
import torch


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
    # 从配置中获取模型名称
    model_name = _extract_model_name_from_path(base_config.inference.weak_model_path)

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

            hidden_states_file = _build_hs_path(task, model_name)

            if not os.path.exists(hidden_states_file):
                print(f"Warning: Hidden states file not found: {hidden_states_file}")
                continue

            datasets = [f"{task}"]

            pipeline.evaluate_complete_pipeline(
                hidden_states_file=hidden_states_file,
                datasets=datasets
            )


def _extract_model_name_from_path(model_path: str) -> str:
    """从模型路径中提取模型名称"""
    return os.path.basename(model_path.rstrip('/'))


def _build_hs_path(task: str, model_name: str = None):
    """构建hidden states文件路径"""
    if model_name is None:
        # 从配置文件中读取 weak_model_path
        config = PipelineConfig.from_yaml()
        model_name = _extract_model_name_from_path(config.inference.weak_model_path)

    base_dir = os.path.join("..", "hs")
    if task.startswith("mmlu_pro_"):
        base_dir = os.path.join(base_dir, "mmlu_pro")
    return os.path.join(base_dir, f"{model_name}_{task}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CoBench Router Evaluation and Training Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mode", type=str, required=True,
                       choices=["get_scores", "get_logits", "get_query_emb",
                                "train", "train_embedding_mlp",
                                "eval_probe", "eval_embedding_mlp", "eval_deberta",
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

    parser.add_argument("--query_embeddings_file", type=str, default=None,
                       help="EmbeddingMLP 训练/评估所需的 query embedding pt 文件路径")
    parser.add_argument("--text_field", type=str, default="instruction",
                       help="生成 query embedding 时使用的文本字段")
    parser.add_argument("--embed_batch_size", type=int, default=64,
                       help="生成 query embedding 的 batch size")
    parser.add_argument("--embedding_checkpoint", type=str, default=None,
                       help="EmbeddingMLP checkpoint 路径（评估或自定义保存名）")

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
                continue

            try:
                generate_logits(config, task=task, task_path=task_path)
                print(f"✅ {task} logits 生成完成")
            except Exception as e:
                print(f"❌ {task} logits 生成失败: {e}")

    # ==================== 模式: get_query_emb ====================
    elif mode == "get_query_emb":
        datasets = args.datasets
        print(f"🎯 生成以下数据集的 query embedding: {datasets}")

        for task in datasets:
            print(f"\n{'='*60}")
            print(f"📊 处理任务: {task}")
            print(f"{'='*60}")

            if task.startswith("mmlu_pro_"):
                task_path = os.path.join("./results/mmlu_pro", f"{task}.jsonl")
            else:
                task_path = os.path.join("./results", f"{task}.jsonl")

            if not os.path.exists(task_path):
                print(f"⚠️  警告: 结果文件 {task_path} 不存在，请先运行 get_scores")
                continue

            try:
                save_path = generate_query_embeddings(
                    task_path,
                    output_dir=config.training.query_embedding_output_dir,
                    batch_size=args.embed_batch_size,
                    text_field=args.text_field
                )
                print(f"✅ {task} query embedding 已保存到 {save_path}")
            except Exception as e:
                print(f"❌ {task} 生成 query embedding 失败: {e}")

    # ==================== 模式: train ====================
    elif mode == "train":
        max_samples = args.max_samples
        save_history = args.save_loss_history

        if config.router.router_type in ["trained_deberta", "deberta"]:
            train_path = config.training.deberta_train_path
            val_path = config.training.deberta_val_path

            if args.datasets:
                try:
                    train_path = prepare_deberta_training_file(config, args.datasets)
                except Exception as e:
                    print(f"❌ DeBERTa 训练数据准备失败: {e}")
                    import traceback
                    traceback.print_exc()
                    exit(1)

            print("🚀 开始训练 DeBERTa Router")
            print(f"📊 训练数据: {train_path}")
            if val_path:
                print(f"📊 验证数据: {val_path}")

            try:
                output_dir = train_deberta_router(config, train_path=train_path, val_path=val_path)
                final_ckpt = Path(output_dir) / f"checkpoint_epoch_{config.training.deberta_epochs}"
                print(f"✅ DeBERTa 训练完成，输出目录: {output_dir}")
                print(f"   期望最终 checkpoint: {final_ckpt}")
            except Exception as e:
                print(f"❌ DeBERTa 训练失败: {e}")
                import traceback
                traceback.print_exc()

        elif config.router.router_type == "embedding_mlp":
            # 使用配置中的 embedding_files 训练 EmbeddingMLP
            files = config.router.embedding_files or []
            if not files:
                print("❌ 配置 router.embedding_files 为空，无法训练 EmbeddingMLP")
                exit(1)
            all_data = []
            for fp in files:
                if not os.path.exists(fp):
                    print(f"⚠️  嵌入文件不存在: {fp}")
                    continue
                data = torch.load(fp, map_location="cpu", weights_only=False)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                print(f"📂 {fp} -> {len(data)} 条样本")
                all_data.extend(data)
            print(f"🔢 已加载 embedding 样本总数: {len(all_data)}，来自 {len(files)} 个文件")
            if len(all_data) == 0:
                print("❌ 未加载到任何样本")
                exit(1)
            if max_samples and len(all_data) > max_samples:
                all_data = random.sample(all_data, max_samples)

            split = int(len(all_data) * 0.8)
            train_data, val_data = all_data[:split], all_data[split:]

            sample = train_data[0]
            sample_emb = sample.get("query_embedding", None)
            if sample_emb is None:
                sample_emb = sample.get("embedding", None)
            if isinstance(sample_emb, torch.Tensor):
                input_dim = sample_emb.shape[-1]
            elif hasattr(sample_emb, "__len__"):
                input_dim = len(sample_emb)
            else:
                raise ValueError("无法解析样本 embedding 维度")

            hidden_dims = config.training.embedding_hidden_dims
            dropout = config.training.embedding_dropout
            epochs = config.training.epochs
            batch_size = config.training.batch_size
            lr = config.training.learning_rate

            # 自定义保存路径：优先 router.checkpoint_path，否则落到 training.embedding_mlp_save_path
            if config.router.checkpoint_path:
                save_path = Path(config.router.checkpoint_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = Path(config.training.embedding_mlp_save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_name = f"embedding_mlp_{Path(files[0]).stem}.pt"
                save_path = save_dir / save_name

            print(f"🚀 训练 EmbeddingMLP")
            print(f"   样本数: {len(all_data)} (train {len(train_data)}, val {len(val_data)})")
            print(f"   hidden_dims={hidden_dims}, dropout={dropout}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
            print(f"   保存到: {save_path}")

            res = train_embedding_mlp_model(
                train_data=train_data,
                val_data=val_data,
                input_dim=input_dim,
                save_path=str(save_path),
                hidden_dims=hidden_dims,
                dropout=dropout,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr
            )
            print(f"✅ EmbeddingMLP 训练完成，最优 val_loss={res['best_val_loss']:.4f}")

        else:
            # 原 probe 训练流程
            datasets = args.datasets 
            probe_types = args.probe_types

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
            # 从配置中获取模型名称
            model_name = _extract_model_name_from_path(config.inference.weak_model_path)

            for probe_type in probe_types:
                print(f"\n{'='*60}")
                print(f" 使用 Probe 类型: {probe_type}")
                print(f"{'='*60}")

                config_copy = copy.deepcopy(config)
                config_copy.router.router_type = "probe"
                config_copy.router.probe_type = probe_type
                # 如果配置中已有 checkpoint_path，使用它；否则需要用户指定
                if not config_copy.router.checkpoint_path:
                    print(f"⚠️  警告: 未指定 checkpoint_path，请在配置文件中设置或使用 --probe_dir")
                    continue

                pipeline_test = RouterEvaluationPipeline(config_copy)

                for task in datasets:
                    print(f"\n📊 评估任务: {task}")

                    hidden_states_file = _build_hs_path(task,model_name)

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

    
    # ==================== 模式: eval_embedding_mlp ====================
    elif mode == "eval_embedding_mlp":
        datasets = args.datasets

        # 优先 CLI 提供的 checkpoint，否则用配置
        ckpt_path = args.embedding_checkpoint or config.router.checkpoint_path
        if not ckpt_path:
            print("❌ eval_embedding_mlp 需要 checkpoint（--embedding_checkpoint 或 config.router.checkpoint_path）")
            exit(1)
        if not os.path.exists(ckpt_path):
            print(f"❌ checkpoint 文件不存在: {ckpt_path}")
            exit(1)

        # 选择 embedding 文件：若 CLI 提供则用 CLI，否则根据 config.router.embedding_files 自动匹配
        def _pick_emb_file(task: str):
            if args.query_embeddings_file:
                return args.query_embeddings_file

            task_clean = task.strip()
            exact_stem = f"{task_clean}_query_embeddings"

            # 1) 若配置了 embedding_files，则尝试精确匹配
            files = config.router.embedding_files or []
            if files:
                for fp in files:
                    if Path(fp).stem == exact_stem:
                        return fp

            # 2) 在目录里自动寻找（默认用 training.query_embedding_output_dir）
            base_dir = getattr(config.training, "query_embedding_output_dir", "query_embeddings_output")
            candidate = Path(base_dir) / f"{exact_stem}.pt"
            if candidate.exists():
                return str(candidate)

            # 3) 如果配置了 embedding_files 但没匹配到，给出可用列表
            available = [Path(fp).stem for fp in files] if files else []
            raise ValueError(
                f"未找到与任务 {task_clean} 匹配的 embedding 文件。"
                f"期望文件: {candidate} 或 {exact_stem}.pt，候选: {available}"
            )

        config_copy = copy.deepcopy(config)
        config_copy.router.router_type = "embedding_mlp"
        config_copy.router.checkpoint_path = ckpt_path

        pipeline_test = RouterEvaluationPipeline(config_copy)

        for task in datasets:
            emb_file = _pick_emb_file(task)
            if not emb_file:
                print(f"❌ 未找到 {task} 对应的 embedding 文件（请在 config.router.embedding_files 配置或用 --query_embeddings_file）")
                continue
            if not os.path.exists(emb_file):
                print(f"❌ embedding 文件不存在: {emb_file}")
                continue

            print(f"\n{'='*60}")
            print(f"📊 评估任务: {task}")
            print(f"   使用 embedding 文件: {emb_file}")
            print(f"{'='*60}")

            try:
                pipeline_test.evaluate_complete_pipeline(
                    hidden_states_file=None,
                    datasets=[task],
                    query_embeddings_file=emb_file
                )
                print(f"✅ {task} 评估完成")
            except Exception as e:
                print(f"❌ {task} 评估失败: {e}")

    # ==================== 模式: eval_deberta ====================
    elif mode == "eval_deberta":
        datasets = args.datasets
        if config.router.router_type not in ["deberta", "trained_deberta"]:
            print("⚠️  当前配置 router_type 不是 deberta/trained_deberta，将继续使用配置中的 router_type。")

        config_copy = copy.deepcopy(config)
        pipeline_test = RouterEvaluationPipeline(config_copy)

        for task in datasets:
            print(f"\n{'='*60}")
            print(f"📊 评估任务: {task}")
            print(f"   Router: {config_copy.router.router_type}")
            print(f"{'='*60}")

            try:
                pipeline_test.evaluate_complete_pipeline(
                    hidden_states_file=None,
                    datasets=[task],
                    query_embeddings_file=None
                )
                print(f"✅ {task} 评估完成")
            except Exception as e:
                print(f"❌ {task} 评估失败: {e}")

    elif mode == "self_based":
        # 从配置中获取模型名称
        model_name = _extract_model_name_from_path(config.inference.weak_model_path)

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
                hidden_states_file = _build_hs_path(task, model_name)

                if not os.path.exists(hidden_states_file):
                    print(f"警告: Hidden states 文件不存在: {hidden_states_file}")
                    continue

                datasets = [task]
                pipeline_test.evaluate_complete_pipeline(
                    hidden_states_file=hidden_states_file,
                    datasets=datasets
                )
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
