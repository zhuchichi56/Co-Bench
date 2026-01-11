from config import PipelineConfig
from pipeline import RouterEvaluationPipeline, get_task_score
from train_router import (
    generate_logits,
    set_random_seed,
    complete_probe_training_pipeline_with_mixed_datasets,
    generate_query_embeddings,
    save_training_history,
    train_embedding_mlp_model,
    train_deberta_router,
    prepare_deberta_training_file,
)
import os
import copy
import json
import argparse
import glob
import re
import random
from pathlib import Path
from datetime import datetime
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr


class RunLogger:
    """Capture stdout/stderr from downstream code into a log file."""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, text: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)

    def run(self, label: str, fn, *args, **kwargs):
        out_buf, err_buf = io.StringIO(), io.StringIO()
        self._append(f"\n\n=== {label} ===\n")
        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                return fn(*args, **kwargs)
        except Exception:
            # Persist captured output and exception traceback.
            self._append(out_buf.getvalue())
            self._append(err_buf.getvalue())
            self._append("\n--- exception ---\n")
            self._append(traceback.format_exc())
            raise
        finally:
            out = out_buf.getvalue()
            err = err_buf.getvalue()
            if out:
                self._append(out)
            if err:
                self._append(err)


def _extract_model_name_from_path(model_path: str) -> str:
    """Extract a model name from a path-like string."""
    return os.path.basename(model_path.rstrip('/'))


def _build_task_path(task: str) -> str:
    """Build the result JSONL path for a task."""
    if task.startswith("mmlu_pro_"):
        task_path = os.path.join("./results/mmlu_pro", f"{task}.jsonl")
    else:
        task_path = os.path.join("./results", f"{task}.jsonl")
    return task_path


def _build_hs_path(task: str, model_name: str):
    """Build the hidden-states path for a task."""
    base_dir = os.path.join("..", "hs")
    if task.startswith("mmlu_pro_"):
        base_dir = os.path.join(base_dir, "mmlu_pro")
    return os.path.join(base_dir, f"{model_name}_{task}.pt")


def prepare_data(
    config: PipelineConfig,
    datasets: list,
    steps: list = None,
    text_field: str = "instruction",
    embed_batch_size: int = 64,
    logger: RunLogger = None,
):
    """Prepare artifacts (scores/logits/embeddings). Returns per-task output paths."""
    if steps is None:
        steps = ["scores", "logits", "embeddings"]

    valid_steps = ["scores", "logits", "embeddings"]
    invalid_steps = [s for s in steps if s not in valid_steps]
    if invalid_steps:
        raise ValueError(f"Invalid steps: {invalid_steps}. Valid: {valid_steps}")

    if logger is None:
        raise ValueError("logger is required")

    outputs: dict[str, dict[str, str]] = {}

    for task in datasets:
        task_out: dict[str, str] = {}

        if "scores" in steps:
            score_path = logger.run("prepare_scores", get_task_score, config, task=task)
            if score_path:
                task_out["scores"] = str(score_path)

        if "logits" in steps:
            task_path = _build_task_path(task)
            if os.path.exists(task_path):
                logits_file = logger.run("prepare_logits", generate_logits, config, task, task_path)
                task_out["logits"] = str(logits_file)

        if "embeddings" in steps:
            task_path = _build_task_path(task)
            if os.path.exists(task_path):
                save_path = logger.run(
                    "prepare_embeddings",
                    generate_query_embeddings,
                    task_path,
                    config.training.query_embedding_output_dir,
                    embed_batch_size,
                    text_field,
                )
                task_out["embeddings"] = str(save_path)

        outputs[task] = task_out

    return outputs


def _get_required_files_for_router(router_type: str, task: str, config: PipelineConfig, 
                                   model_name: str = None, query_embeddings_file: str = None):
    """Return (hidden_states_file, query_embeddings_file) for the given router/task."""
    hidden_states_file = None
    emb_file = query_embeddings_file
    
    hs_required_types = {
        "probe",
        "coe",
        "max_logits",
        "top10_variance",
        "entropy",
        "confidence_margin",
        "semantic_entropy",
        "self_questioning",
    }

    if router_type in hs_required_types:
        if model_name is None:
            model_name = _extract_model_name_from_path(config.inference.weak_model_path)
        hidden_states_file = _build_hs_path(task, model_name)
    
    elif router_type == "embedding_mlp":
        if not emb_file:
            task_clean = task.strip()
            exact_stem = f"{task_clean}_query_embeddings"
            exact_name = f"{exact_stem}.pt"
            
            # 1) If a directory is provided, look up by filename (hs-like behavior).
            emb_dir = getattr(config.router, "embedding_dir", None)
            if emb_dir:
                candidate = Path(emb_dir) / exact_name
                if candidate.exists():
                    emb_file = str(candidate)

            # 2) Backward-compatible: allow embedding_files to include files or directories.
            if not emb_file:
                files = config.router.embedding_files or []
                for fp in files:
                    p = Path(fp)
                    if p.is_dir():
                        candidate = p / exact_name
                        if candidate.exists():
                            emb_file = str(candidate)
                            break
                    else:
                        if p.stem == exact_stem and p.exists():
                            emb_file = str(p)
                            break
            
            if not emb_file:
                base_dir = getattr(config.training, "query_embedding_output_dir", "query_embeddings_output")
                candidate = Path(base_dir) / f"{exact_stem}.pt"
                if candidate.exists():
                    emb_file = str(candidate)

    
    return hidden_states_file, emb_file


def evaluate_router(config: PipelineConfig, datasets: list, router_type: str = None, logger: RunLogger = None):
    """Evaluate according to config.router.router_type (and its special batch modes)."""
    if router_type is None:
        router_type = config.router.router_type

    if logger is None:
        raise ValueError("logger is required")
    
    model_name = _extract_model_name_from_path(config.inference.weak_model_path)
    probe_dir = getattr(config, "probe_dir", None)
    probe_types = getattr(config.training, "probe_types", None)
    metric_results_dir = getattr(config, "metric_results_dir", "metric_results")
    
    if router_type == "probe" and probe_dir:
        probe_files = sorted(glob.glob(f"{probe_dir}/*.pt"))
        if not probe_files:
            return
        probe_configs = []
        
        for pf in probe_files:
            filename = os.path.basename(pf)
            m = re.search(r'.*?_(?:train_)?([^_]+)\.pt$', filename)
            if m:
                detected_probe_type = m.group(1)
                if probe_types and detected_probe_type not in probe_types:
                    continue
                
                probe_configs.append({
                    "checkpoint_path": pf,
                    "probe_type": detected_probe_type,
                    "metric_results_dir": metric_results_dir,
                })
        
        if probe_configs:
            for probe_config in probe_configs:
                config_copy = copy.deepcopy(config)
                config_copy.router.checkpoint_path = probe_config["checkpoint_path"]
                config_copy.router.probe_type = probe_config["probe_type"]
                config_copy.metric_results_dir = probe_config["metric_results_dir"]
                pipeline = RouterEvaluationPipeline(config_copy)

                for task in datasets:
                    hs_file = _build_hs_path(task, model_name)
                    if not os.path.exists(hs_file):
                        continue
                    logger.run(
                        f"eval_probe_dir task={task} probe_type={probe_config['probe_type']}",
                        pipeline.evaluate_complete_pipeline,
                        hs_file,
                        [task],
                        None,
                    )
        return
    
    if router_type == "logits_based_routers":
        router_types = ["max_logits", "top10_variance", "coe", "entropy", "confidence_margin"]
        for rt in router_types:
            config_copy = copy.deepcopy(config)
            config_copy.router.router_type = rt
            base_dir = metric_results_dir
            config_copy.metric_results_dir = str(Path(base_dir) / "base" / rt)
            evaluate_router(config_copy, datasets, router_type=rt, logger=logger)
        return
    
    if router_type == "self_based":
        strategies = [
            {"name": "semantic_entropy", "metric_results_dir": "metric_results/base/semantic_entropy", "num_samples": 5},
            {"name": "self_questioning", "metric_results_dir": "metric_results/base/self_questioning", "num_samples": 8},
        ]
        for strat in strategies:
            config_copy = copy.deepcopy(config)
            base_dir = metric_results_dir
            config_copy.metric_results_dir = str(Path(base_dir) / "base" / strat["name"])
            config_copy.router.router_type = strat["name"]
            config_copy.router.model_path = None
            config_copy.router.num_samples = strat["num_samples"]
            evaluate_router(config_copy, datasets, router_type=strat["name"], logger=logger)
        return
    
    config_copy = copy.deepcopy(config)
    config_copy.router.router_type = router_type
    config_copy.metric_results_dir = metric_results_dir
    
    if router_type == "probe" and probe_types and len(probe_types) > 1:
        if not config_copy.router.checkpoint_path:
            return
        
        for probe_type in probe_types:
            config_probe = copy.deepcopy(config_copy)
            config_probe.router.probe_type = probe_type
            
            pipeline_probe = RouterEvaluationPipeline(config_probe)
            for task in datasets:
                hidden_states_file, _ = _get_required_files_for_router(
                    router_type, task, config_probe, model_name, None
                )
                if hidden_states_file and not os.path.exists(hidden_states_file):
                    continue
                logger.run(
                    f"eval task={task} router=probe probe_type={probe_type}",
                    pipeline_probe.evaluate_complete_pipeline,
                    hidden_states_file,
                    [task],
                    None,
                )
        return
    
    if router_type == "probe" and probe_types and len(probe_types) == 1:
        config_copy.router.probe_type = probe_types[0]
    
    if router_type == "probe" and not config_copy.router.checkpoint_path:
        return
    
    pipeline = RouterEvaluationPipeline(config_copy)
    
    for task in datasets:
        hidden_states_file, emb_file = _get_required_files_for_router(
            router_type, task, config_copy, model_name, None
        )
        
        if hidden_states_file and not os.path.exists(hidden_states_file):
            continue
        
        if router_type == "embedding_mlp":
            if not emb_file:
                continue
            if not os.path.exists(emb_file):
                continue

        logger.run(
            f"eval task={task} router={router_type}",
            pipeline.evaluate_complete_pipeline,
            hidden_states_file,
            [task],
            emb_file,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CoBench Router Evaluation and Training Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mode", type=str, required=True,
                       choices=["prepare", "train", "eval"],
                       help="Mode: prepare | train | eval")

    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                       help="Datasets to run (space-separated)")

    args = parser.parse_args()
    mode = args.mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = RunLogger(log_file=str(Path("logs") / f"main_{mode}_{timestamp}.log"))

    config = logger.run("load_config", PipelineConfig.from_yaml)
    if config.inference.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.inference.cuda_visible_devices
    set_random_seed(getattr(config.training, "seed", 42))

    print(f"mode={mode}")
    print(f"log_file={logger.log_file}")

    # prepare
    if mode == "prepare":
        if not args.datasets:
            raise ValueError("prepare requires --datasets")
        steps = getattr(config, "prepare_steps", ["scores", "logits", "embeddings"])
        text_field = getattr(config, "prepare_text_field", "instruction")
        embed_bs = getattr(config, "prepare_embed_batch_size", 64)
        outputs = prepare_data(
            config=config,
            datasets=args.datasets,
            steps=steps,
            text_field=text_field,
            embed_batch_size=embed_bs,
            logger=logger,
        )
        print(f"saved_prepare_outputs={json.dumps(outputs, ensure_ascii=False)}")

    # train
    elif mode == "train":
        max_samples = getattr(config.training, "max_samples", 4000)
        save_history = getattr(config.training, "save_loss_history", False)

        if config.router.router_type in ["trained_deberta", "deberta"]:
            train_path = config.training.deberta_train_path
            val_path = config.training.deberta_val_path

            if args.datasets:
                train_path = logger.run(
                    "prepare_deberta_training_file",
                    prepare_deberta_training_file,
                    config,
                    args.datasets,
                )

            output_dir = logger.run("train_deberta_router", train_deberta_router, config, train_path, val_path)
            final_ckpt = Path(output_dir) / f"checkpoint_epoch_{config.training.deberta_epochs}"
            print(f"saved_model_dir={output_dir}")
            print(f"saved_checkpoint={final_ckpt}")

        elif config.router.router_type == "embedding_mlp":
            files = config.router.embedding_files or []
            if not files:
                raise ValueError("router.embedding_files is empty; cannot train embedding_mlp")
            all_data = []
            for fp in files:
                if not os.path.exists(fp):
                    continue
                import torch
                data = logger.run(f"load_embedding_file {fp}", torch.load, fp, "cpu", False)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                all_data.extend(data)
            if len(all_data) == 0:
                raise ValueError("No samples loaded from embedding_files")
            if max_samples and len(all_data) > max_samples:
                all_data = random.sample(all_data, max_samples)

            split = int(len(all_data) * 0.8)
            train_data, val_data = all_data[:split], all_data[split:]

            sample = train_data[0]
            sample_emb = sample.get("query_embedding", None)
            if sample_emb is None:
                sample_emb = sample.get("embedding", None)
            import torch
            if isinstance(sample_emb, torch.Tensor):
                input_dim = sample_emb.shape[-1]
            elif hasattr(sample_emb, "__len__"):
                input_dim = len(sample_emb)
            else:
                raise ValueError("Cannot infer embedding dimension from sample")

            hidden_dims = config.training.embedding_hidden_dims
            dropout = config.training.embedding_dropout
            epochs = config.training.epochs
            batch_size = config.training.batch_size
            lr = config.training.learning_rate

            # Save path: prefer router.checkpoint_path; otherwise use training.embedding_mlp_save_path.
            if config.router.checkpoint_path:
                save_path = Path(config.router.checkpoint_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = Path(config.training.embedding_mlp_save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_name = f"embedding_mlp_{Path(files[0]).stem}.pt"
                save_path = save_dir / save_name

            logger.run(
                "train_embedding_mlp_model",
                train_embedding_mlp_model,
                train_data,
                val_data,
                input_dim,
                str(save_path),
                hidden_dims,
                dropout,
                epochs,
                batch_size,
                lr,
            )
            print(f"saved_checkpoint={save_path}")

        else:
            datasets = args.datasets 
            probe_types = getattr(config.training, "probe_types", None) or ["hs_last_mlp", "mean", "max", "coe_dual_mlp"]
            saved_models = []
            saved_histories = []
            for probe_type in probe_types:
                config.router.probe_type = probe_type
                history = logger.run(
                    f"train_probe probe_type={probe_type}",
                    complete_probe_training_pipeline_with_mixed_datasets,
                    config,
                    datasets,
                    "balanced",
                    max_samples,
                    None,
                    None,
                    False,
                )
                if isinstance(history, dict) and history.get("model_path"):
                    saved_models.append(history["model_path"])
                if save_history and history:
                    saved_histories.append(save_training_history(history, probe_type, datasets, max_samples))

            if saved_models:
                print(f"saved_models={json.dumps(saved_models, ensure_ascii=False)}")
            if saved_histories:
                print(f"saved_training_histories={json.dumps(saved_histories, ensure_ascii=False)}")

    # eval
    elif mode == "eval":
        if not args.datasets:
            raise ValueError("eval requires --datasets")
        evaluate_router(config=config, datasets=args.datasets, logger=logger)
        print(f"saved_metric_results_dir={config.metric_results_dir}")

    else:
        raise ValueError(f"Unknown mode: {mode}")
