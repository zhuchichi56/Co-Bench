import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import random
from tqdm import tqdm
try:
    import fire  # type: ignore
except ModuleNotFoundError:
    fire = None
from config import PipelineConfig
import os
import torch.nn.functional as F
from router import MLPProbe, ConvProbe, MeanProbe, MaxProbe, ZScoreNormalizer, DynamicFusionProbe, EmbeddingMLPNet
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing as mp
import gc
import sys
from pathlib import Path
from datetime import datetime
from utils.train_deberta import DeBERTaTrainer, QuestionDataset, TrainingConfig as DebertaTrainingConfig


def save_training_history(
    history: Dict[str, Any],
    probe_type: str,
    task_list: List[str],
    max_samples: Optional[int] = None,
    save_dir: str = "probe_save/loss",
) -> str:
    """
    Save probe training history (loss/accuracy/auroc curves, etc.) to a JSON file.

    This is optional metadata for experiment tracking; it is not required for training/evaluation.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks_str = "_".join(task_list)
    filename = f"{tasks_str}_{probe_type}_{timestamp}.json"
    filepath = save_path / filename

    training_results = history.get("training_results", {}) if isinstance(history, dict) else {}

    save_data = {
        "probe_type": probe_type,
        "tasks": task_list,
        "datasets": task_list,
        "max_samples": max_samples,
        "timestamp": timestamp,
        "initial_lr": training_results.get("initial_lr", 0.0),
        "best_val_loss": training_results.get("best_val_loss", float("inf")),
        "best_val_acc": training_results.get("best_val_acc", 0.0),
        "best_val_auroc": training_results.get("best_val_auroc", 0.0),
        "epochs": len(training_results.get("train_losses", []) or []),
        "train_losses": training_results.get("train_losses", []) or [],
        "val_losses": training_results.get("val_losses", []) or [],
        "val_accuracies": training_results.get("val_accuracies", []) or [],
        "val_aurocs": training_results.get("val_aurocs", []) or [],
        "learning_rates": training_results.get("learning_rates", []) or [],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    return str(filepath)


class ProbeDataset(Dataset):
    def __init__(self, data: List[Dict], probe_type: str, normalizer: Optional[ZScoreNormalizer] = None):
        self.data = data
        self.probe_type = probe_type
        self.normalizer = normalizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        hidden_states = torch.tensor(item.get("hidden_states", []), dtype=torch.float32)
        label = torch.tensor(item.get("acc_label", 0), dtype=torch.float32)

        if not hasattr(self, '_debug_checked'):
            self._debug_checked = True
            if "acc_label" not in item:
                print(f"⚠️ Warning: 'acc_label' not found in item {idx}. Item keys: {list(item.keys())}")
                print(f"   Item sample: {item}")

        # Feature extraction based on probe type
        if self.probe_type == "hs_last_mlp":
            features = hidden_states[-1]
        elif self.probe_type in ["coe_dual_mlp", "coe_c_scalar", "coe_r_scalar"]:
            mag_features = []
            angle_features = []
            for i in range(hidden_states.shape[0] - 1):
                h_curr, h_next = hidden_states[i], hidden_states[i+1]
                mag_features.append(torch.norm(h_curr, p=2) + torch.norm(h_next, p=2))
                angle_features.append(F.cosine_similarity(h_curr, h_next, dim=0))
            features = torch.cat([torch.stack(mag_features), torch.stack(angle_features)], dim=0)
        else:
            features = hidden_states

        if self.normalizer and self.probe_type not in ["pca_conv", "mean", "max", "hs_mlp", "hs_last_mlp", "dynamic_softmax", "dynamic_dirichlet"]:
            features = self.normalizer(features)

        return features, label

class EmbeddingDataset(Dataset):
    """Dataset for EmbeddingMLP router (uses query-level embeddings)."""

    def __init__(self, data: List[Dict], normalizer: Optional[ZScoreNormalizer] = None):
        self.data = data
        self.normalizer = normalizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        emb = item.get("query_embedding", None)
        if emb is None:
            emb = item.get("embedding", None)
        if emb is None:
            raise ValueError(f"Missing query_embedding in sample idx={idx}")
        emb = torch.tensor(emb, dtype=torch.float32)
        label = torch.tensor(item.get("acc_label", 0), dtype=torch.float32)
        if self.normalizer:
            emb = self.normalizer(emb)
        return emb, label


class ProbeTrainer:
    PROBE_CLASSES = {
        "hs_last_mlp": MLPProbe, "hs_mlp": MLPProbe, "coe_dual_mlp": MLPProbe,
        "coe_c_scalar": MLPProbe, "coe_r_scalar": MLPProbe, "pca_conv": ConvProbe,
        "mean": MeanProbe, "max": MaxProbe,
        "dynamic_softmax": DynamicFusionProbe, "dynamic_dirichlet": DynamicFusionProbe
    }

    def __init__(self, probe_type: str, device: Optional[str] = None, probe_config: Optional[Dict] = None):
        self.probe_type = probe_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.normalizer = None
        self.probe_config = probe_config or {}

    def build_normalizer(self, train_dataset) -> Optional[ZScoreNormalizer]:
        if self.probe_type in ["pca_conv", "mean", "max", "hs_mlp", "hs_last_mlp", "dynamic_softmax", "dynamic_dirichlet"]:
            return None

        features = []
        for feat, _ in train_dataset:
            if isinstance(feat, torch.Tensor):
                features.append(feat.unsqueeze(0))
            elif isinstance(feat, np.ndarray):
                features.append(torch.from_numpy(feat).unsqueeze(0))

        X = torch.cat(features, dim=0).float()
        mask = torch.isfinite(X).all(dim=1)
        X = X[mask]
        mu = X.mean(dim=0)
        std = X.std(dim=0).clamp_min(1e-6)
        return ZScoreNormalizer(mu, std)

    def get_input_dim(self, sample) -> int:
        hidden_states = sample[0]
        if self.probe_type in ["hs_last_mlp", "hs_mlp"]:
            return hidden_states.shape[0]
        elif self.probe_type in ["coe_dual_mlp", "coe_c_scalar", "coe_r_scalar"]:

            return hidden_states.shape[0]
        elif self.probe_type in ["dynamic_softmax", "dynamic_dirichlet"]:
            # Input shape: [batch_size, num_layers, hidden_dim] -> return hidden_dim
            return hidden_states.shape[2] if len(hidden_states.shape) > 2 else hidden_states.shape[1]
        return hidden_states.shape[1]
        

    def create_model(self, input_dim: int, output_dim: int = 1) -> nn.Module:
        probe_class = self.PROBE_CLASSES[self.probe_type]

        # Build kwargs based on probe type
        kwargs = {}

        if self.probe_type in ["hs_last_mlp", "hs_mlp", "coe_dual_mlp", "coe_c_scalar", "coe_r_scalar", "mean", "max"]:
            # MLP probes (including mean/max probes that support MLP structure)
            if "mlp_hidden_dims" in self.probe_config and self.probe_config["mlp_hidden_dims"]:
                kwargs["hidden_dims"] = self.probe_config["mlp_hidden_dims"]
            if "mlp_dropout" in self.probe_config:
                kwargs["dropout"] = self.probe_config["mlp_dropout"]

        elif self.probe_type == "pca_conv":
            # Conv probe
            if "conv_channels" in self.probe_config:
                kwargs["conv_channels"] = self.probe_config["conv_channels"]
            if "conv_kernel_size" in self.probe_config:
                kwargs["kernel_size"] = self.probe_config["conv_kernel_size"]

        elif self.probe_type in ["dynamic_softmax", "dynamic_dirichlet"]:
            # DynamicFusionProbe requires num_layers.
            # Read from probe_config; default to 32 (typical Llama transformer layers) if unset.
            num_layers = self.probe_config.get("num_layers", 32)
            probe_subtype = "softmax" if self.probe_type == "dynamic_softmax" else "dirichlet"
            return probe_class(input_dim, num_layers, output_dim, probe_type=probe_subtype)

        return probe_class(input_dim, output_dim, **kwargs)

    def train(self, train_data: List[Dict], val_data: List[Dict], epochs: int = 50,
              batch_size: int = 32, lr: float = 1e-4, save_path: Optional[str] = None) -> Dict:

        # Setup datasets and normalizer
        temp_dataset = ProbeDataset(train_data, self.probe_type, normalizer=None)
        self.normalizer = self.build_normalizer(temp_dataset)
        train_dataset = ProbeDataset(train_data, self.probe_type, self.normalizer)
        val_dataset = ProbeDataset(val_data, self.probe_type, self.normalizer)

        # Setup model and training
        num_gpus = torch.cuda.device_count()
        effective_batch_size = batch_size * max(1, num_gpus)
        input_dim = self.get_input_dim(train_dataset[0])

        if self.probe_type in ["dynamic_softmax", "dynamic_dirichlet", "mean"]:
            if "num_layers" not in self.probe_config or self.probe_config.get("num_layers") is None:
                sample_feat = train_dataset[0][0]  # [num_layers, hidden_dim]
                inferred_num_layers = int(sample_feat.shape[0])
                self.probe_config["num_layers"] = inferred_num_layers
                print(f"🧠 [DynamicFusionProbe] Inferred num_layers={inferred_num_layers} from training data")

        self.model = self.create_model(input_dim).to(self.device)

        # Print detailed model structure
        print("\n" + "="*80)
        print("Model Architecture:")
        print(self.model)
        print("\n" + "-"*80)
        
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)
            print(f"🔧 Using DataParallel with {num_gpus} GPUs")

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True,
                                num_workers=min(8, num_gpus * 2), pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False,
                              num_workers=min(8, num_gpus * 2), pin_memory=True)

        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_auroc = 0.0
        train_losses, val_losses = [], []
        val_accuracies = []
        val_aurocs = []
        learning_rates = []
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)

                if self.probe_type == "pca_conv":
                    batch_features = batch_features.permute(0, 2, 1).contiguous()

                optimizer.zero_grad()
                outputs = self.model(batch_features).squeeze(-1)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for batch_features, batch_labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_labels = batch_labels.to(self.device, non_blocking=True)

                    if self.probe_type == "pca_conv":
                        batch_features = batch_features.permute(0, 2, 1).contiguous()

                    outputs = self.model(batch_features).squeeze(-1)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    probs = torch.sigmoid(outputs)
                    predictions = probs > 0.5
                    correct += (predictions == batch_labels.bool()).sum().item()
                    total += batch_labels.size(0)

                    # Collect for AUROC calculation
                    all_probs.extend(probs.cpu().numpy().tolist())
                    all_labels.extend(batch_labels.cpu().numpy().tolist())

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = correct / total

            # Calculate AUROC
            try:
                from sklearn.metrics import roc_auc_score
                val_auroc = roc_auc_score(all_labels, all_probs)
            except:
                val_auroc = 0.0

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_aurocs.append(val_auroc)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_accuracy:.4f}, Val AUROC={val_auroc:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_accuracy
                best_val_auroc = val_auroc
                if save_path:
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    self._save_checkpoint(save_path, input_dim, model_to_save)
                    print(f"💾 New best model saved! Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}, Val AUROC: {best_val_auroc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 8 and epoch > 10:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_aurocs': val_aurocs,
            'learning_rates': learning_rates,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_val_auroc': best_val_auroc,
            'initial_lr': lr
        }

    def _save_checkpoint(self, save_path: str, input_dim: int, model_to_save=None):
        try:
            if model_to_save is None:
                model_to_save = self.model

            model_state_dict = model_to_save.module.state_dict() if hasattr(model_to_save, 'module') else model_to_save.state_dict()

            checkpoint = {
                'model_state_dict': model_state_dict,
                'metadata': {'probe_type': self.probe_type, 'input_dim': input_dim, 'output_dim': 1, 'device': self.device}
            }


            if self.probe_type in ["dynamic_softmax", "dynamic_dirichlet"]:
                num_layers = self.probe_config.get("num_layers", None)
                checkpoint['num_layers'] = num_layers
            if self.normalizer:
                checkpoint['normalizer'] = {'mu': self.normalizer.mu, 'std': self.normalizer.std}

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f"💾 Model saved to {save_path}")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            raise


def _process_data_batch(model, tokenizer, batch_data, device):
    """Process a batch of data through the model"""
    instructions = [item.get("instruction", "") for item in batch_data]
    scores = [item.get("score", 0.0) for item in batch_data]

    inputs = tokenizer(instructions, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        attn_mask = inputs.get("attention_mask", None)

        results = []
        for batch_idx in range(len(instructions)):
            hs_list = []
            for layer_states in outputs.hidden_states:
                layer_state = layer_states[batch_idx:batch_idx+1]
                if attn_mask is not None:
                    mask = attn_mask[batch_idx:batch_idx+1].unsqueeze(-1).to(torch.bool)
                    ls = layer_state.to(torch.float32)
                    ls = torch.where(mask, ls, torch.zeros_like(ls))
                    count = mask.sum(dim=1).clamp(min=1).to(ls.dtype)
                    pooled = ls.sum(dim=1) / count
                else:
                    pooled = layer_state.mean(dim=1)
                hs_list.append(pooled.squeeze(0))

            hidden_states = torch.stack(hs_list, dim=0).to(torch.float32).cpu().numpy()
            results.append((hidden_states, scores[batch_idx]))

        return results


def _process_model_single_gpu(model_path: str, data_list: List[dict], dataset_path: str,
                             output_dir: Path, batch_size: int):
    """Process model on single GPU"""
    print(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    device = next(model.parameters()).device

    all_results = []
    for i in tqdm(range(0, len(data_list), batch_size), desc=f"Batches ({Path(model_path).name})"):
        batch_data = data_list[i:i + batch_size]
        batch_results = _process_data_batch(model, tokenizer, batch_data, device)
        all_results.extend(batch_results)

    # Save results
    model_name = Path(model_path).name
    task_name = Path(dataset_path).stem
    output_path = output_dir / f"{model_name}_{task_name}.pt"
    if task_name.startswith("mmlu_pro_"):
        output_dir = os.path.join(output_dir,"mmlu_pro")
        output_path = output_dir / f"{model_name}_{task_name}.pt"
    torch.save(all_results, output_path)
    print(f"Saved {len(all_results)} samples to {output_path}")

    del model, tokenizer
    torch.cuda.empty_cache()


def _worker_process_data(gpu_id: int, model_path: str, data_chunk: List[dict],
                        chunk_start_idx: int, result_queue, batch_size: int):
    """Worker process for multi-GPU processing"""
    torch.cuda.set_device(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map={"": f"cuda:{gpu_id}"}, trust_remote_code=True)
    model.eval()

    results = []
    for i in range(0, len(data_chunk), batch_size):
        batch_data = data_chunk[i:i + batch_size]
        batch_results = _process_data_batch(model, tokenizer, batch_data, f"cuda:{gpu_id}")

        # Add original indices
        for j, (hidden_states, score) in enumerate(batch_results):
            original_idx = chunk_start_idx + i + j
            results.append((original_idx, hidden_states, score))

    result_queue.put((gpu_id, results))
    del model, tokenizer
    torch.cuda.empty_cache()


def _process_model_multi_gpu(model_path: str, data_list: List[dict], dataset_path: str,
                           output_dir: Path, batch_size: int, num_gpus: int):
    """Multi-GPU processing with order preservation"""
    print(f"Loading model from {model_path} on {num_gpus} GPUs")

    # Split data into chunks
    chunk_size = (len(data_list) + num_gpus - 1) // num_gpus
    data_chunks = []
    chunk_start_indices = []

    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data_list))
        if start_idx < len(data_list):
            data_chunks.append(data_list[start_idx:end_idx])
            chunk_start_indices.append(start_idx)

    # Process with multiprocessing
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    processes = []

    for gpu_id in range(len(data_chunks)):
        p = ctx.Process(target=_worker_process_data, args=(
            gpu_id, model_path, data_chunks[gpu_id], chunk_start_indices[gpu_id], result_queue, batch_size))
        p.start()
        processes.append(p)

    # Collect and sort results
    gpu_results = {}
    for _ in range(len(processes)):
        gpu_id, results = result_queue.get()
        gpu_results[gpu_id] = results

    for p in processes:
        p.join()

    # Reconstruct ordered results
    all_results_with_idx = []
    for gpu_id in range(len(data_chunks)):
        if gpu_id in gpu_results:
            all_results_with_idx.extend(gpu_results[gpu_id])

    all_results_with_idx.sort(key=lambda x: x[0])
    all_results = [(hidden_states, score) for _, hidden_states, score in all_results_with_idx]

    # Save results
    model_name = Path(model_path).name
    task_name = Path(dataset_path).stem
    output_path = output_dir / f"{model_name}_{task_name}.pt"
    torch.save(all_results, output_path)
    print(f"Saved {len(all_results)} samples to {output_path}")

    gc.collect()
    torch.cuda.empty_cache()


def train_probe_model(train_data: List[Dict], val_data: List[Dict], probe_type: str,
                     save_path: str, probe_config: Optional[Dict] = None, **kwargs) -> Dict:
    trainer = ProbeTrainer(probe_type, probe_config=probe_config)
    return trainer.train(train_data, val_data, save_path=save_path, **kwargs)


def train_embedding_mlp_model(train_data: List[Dict], val_data: List[Dict],
                              input_dim: int,
                              save_path: str,
                              hidden_dims: Optional[List[int]] = None,
                              dropout: float = 0.1,
                              epochs: int = 50,
                              batch_size: int = 32,
                              lr: float = 1e-3) -> Dict:
    """
    Train an EmbeddingMLP router.
    train/val data should contain fields: query_embedding, acc_label.
    """
    trainer = EmbeddingMLPTrainer(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    return trainer.train(train_data, val_data, epochs=epochs, batch_size=batch_size, lr=lr, save_path=save_path)


def train_deberta_router(config: PipelineConfig, train_path: str, val_path: Optional[str] = None) -> str:
    """Train a DeBERTa router model using config.training settings."""
    if not train_path:
        raise ValueError("deberta_train_path is required for DeBERTa training.")
    if not Path(train_path).exists():
        raise FileNotFoundError(f"DeBERTa train file not found: {train_path}")
    if val_path and not Path(val_path).exists():
        raise FileNotFoundError(f"DeBERTa val file not found: {val_path}")

    train_cfg = DebertaTrainingConfig()
    train_cfg.model_name = config.training.deberta_model_name
    train_cfg.num_labels = config.training.deberta_num_labels
    train_cfg.max_length = config.training.deberta_max_length
    train_cfg.batch_size = config.training.deberta_batch_size
    train_cfg.learning_rate = config.training.deberta_learning_rate
    train_cfg.weight_decay = config.training.deberta_weight_decay
    train_cfg.epochs = config.training.deberta_epochs
    train_cfg.output_dir = config.training.deberta_output_dir

    trainer = DeBERTaTrainer(train_cfg)
    trainer.setup_distributed()

    train_dataset = QuestionDataset(train_path, trainer.tokenizer, max_length=train_cfg.max_length)
    val_dataset = QuestionDataset(val_path, trainer.tokenizer, max_length=train_cfg.max_length) if val_path else None

    if trainer.rank == 0:
        print(f"📊 DeBERTa train size: {len(train_dataset)}")
        if val_dataset:
            print(f"📊 DeBERTa val size: {len(val_dataset)}")

    trainer.train(train_dataset, val_dataset)

    if trainer.world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()

    return train_cfg.output_dir


# -------------------------------------------------------------------------
# Query embedding generation (from LLMRouter Longformer encoder)
# -------------------------------------------------------------------------


def _import_longformer_embedding():
    """Lazy import Longformer embedding function from local LLMRouter repo."""
    repo_root = Path(__file__).resolve().parents[2] / "LLMRouter"
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from llmrouter.utils.embeddings import get_longformer_embedding  # type: ignore
    return get_longformer_embedding


def generate_query_embeddings(dataset_path: str,
                              output_dir: str = "query_embeddings_output",
                              batch_size: int = 64,
                              text_field: str = "instruction") -> Path:
    """
    Generate Longformer embeddings for a dataset and save as .pt

    Args:
        dataset_path: JSONL file with a text field (default: "instruction").
        output_dir: directory to save embeddings.
        batch_size: embedding batch size.
        text_field: key in JSON objects that holds the text to embed.

    Returns:
        Path to saved .pt file containing a list of dicts: {"query_embedding": tensor}
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Load data
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    get_longformer_embedding = _import_longformer_embedding()

    embeddings = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Embedding"):
        batch = samples[i:i + batch_size]
        texts = [item.get(text_field, "") for item in batch]
        scores = [float(item.get("score", 0.0)) for item in batch]  # align with generate_logits acc_label
        with torch.no_grad():
            batch_emb = get_longformer_embedding(texts)
        # get_longformer_embedding returns Tensor for list input
        for emb, sc in zip(batch_emb, scores):
            embeddings.append({
                "query_embedding": emb.cpu(),
                "acc_label": sc
            })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{dataset_path.stem}_query_embeddings.pt"
    torch.save({"data": embeddings}, save_path)
    return save_path


def prepare_deberta_training_file(
    config: PipelineConfig,
    datasets: List[str],
    save_path: Optional[str] = None,
    default_llm_id: Optional[str] = None,
) -> str:
    """
    Build a DeBERTa training JSONL file from results/*.jsonl.
    Each output line: {"question": ..., "label": 0/1}
    """
    if not datasets:
        raise ValueError("datasets is empty for DeBERTa training.")

    base_dir = Path(config.output_dir)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir


    def _resolve_results_file(task: str) -> Path:
        task_path = Path(task)
        if task_path.exists():
            return task_path
        if task.endswith(".jsonl"):
            candidate = base_dir / task
            if candidate.exists():
                return candidate
        if task.startswith("mmlu_pro_"):
            candidate = base_dir / "mmlu_pro" / f"{task}.jsonl"
            if candidate.exists():
                return candidate
        candidate = base_dir / f"{task}.jsonl"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Results file not found for dataset: {task}")

    if save_path is None:
        safe_name = "_".join([Path(d).stem.replace(".jsonl", "") for d in datasets])
        save_dir = Path(config.training.deberta_output_dir)
        if not save_dir.is_absolute():
            save_dir = Path.cwd() / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"deberta_train_{safe_name}.jsonl")

    total_written = 0
    with open(save_path, "w", encoding="utf-8") as out_f:
        for task in datasets:
            fp = _resolve_results_file(task)
            with open(fp, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    question = item.get("question") or item.get("instruction") or item.get("prompt") or ""
                    if not question:
                        continue
                    label = item.get("label", None)
                    if label is None:
                        score = float(item.get("score", 0.0))
                        label = 1 if score > 0.5 else 0
                    out_f.write(json.dumps({"question": question, "label": int(label)}) + "\n")
                    total_written += 1

    print(f"🧱 DeBERTa training data generated: {save_path} (samples={total_written})")
    return save_path


class EmbeddingMLPTrainer:
    """Trainer for EmbeddingMLP router."""

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.1,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmbeddingMLPNet(input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)
        self.dropout = dropout
        self.normalizer = None

    @staticmethod
    def build_normalizer(dataset: EmbeddingDataset) -> Optional[ZScoreNormalizer]:
        feats = []
        for emb, _ in dataset:
            feats.append(emb.unsqueeze(0))
        X = torch.cat(feats, dim=0)
        mu = X.mean(dim=0)
        std = X.std(dim=0).clamp_min(1e-6)
        return ZScoreNormalizer(mu, std)

    def train(self, train_data: List[Dict], val_data: List[Dict], epochs: int = 50,
              batch_size: int = 32, lr: float = 1e-3, save_path: Optional[str] = None,
              metadata: Optional[Dict[str, Any]] = None) -> Dict:

        # Build datasets
        temp_ds = EmbeddingDataset(train_data, normalizer=None)
        self.normalizer = self.build_normalizer(temp_ds)
        train_ds = EmbeddingDataset(train_data, normalizer=self.normalizer)
        val_ds = EmbeddingDataset(val_data, normalizer=self.normalizer)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for emb, label in train_loader:
                emb = emb.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(emb).squeeze(-1)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * emb.size(0)

            train_loss /= len(train_ds)

            # validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for emb, label in val_loader:
                    emb = emb.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)
                    logits = self.model(emb).squeeze(-1)
                    loss = criterion(logits, label)
                    val_loss += loss.item() * emb.size(0)
            val_loss /= len(val_ds)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"[EmbeddingMLP] Epoch {epoch+1}/{epochs} - train_loss {train_loss:.4f} val_loss {val_loss:.4f}")

            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                payload = {
                    "model_state_dict": self.model.state_dict(),
                    "metadata": {
                        "input_dim": self.model.net[0].in_features,
                        "hidden_dims": [m.out_features for m in self.model.net if isinstance(m, nn.Linear)][:-1],
                        "dropout": self.dropout
                    }
                }
                if metadata:
                    payload["metadata"].update(metadata)
                torch.save(payload, save_path)
                print(f"[EmbeddingMLP] Saved checkpoint to {save_path}")

        return {
            "best_val_loss": best_val_loss,
            "history": history,
            "model_path": save_path
        }


def load_training_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    data_path = Path(data_dir)
    train_files = list(data_path.glob("**/train*.pt"))
    val_files = list(data_path.glob("**/val*.pt"))

    train_data, val_data = [], []

    for file_path in train_files:
        data = torch.load(file_path, map_location="cpu")
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        train_data.extend(data)

    for file_path in val_files:
        data = torch.load(file_path, map_location="cpu")
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        val_data.extend(data)

    return train_data, val_data


def get_available_probe_types() -> List[str]:
    return list(ProbeTrainer.PROBE_CLASSES.keys())


def generate_logits_for_models(model_paths: List[str], dataset_path: str, output_dir: str = "logits_output",
                              batch_size: int = 8, num_gpus: int = 1):
    """Generate logits and hidden states for given models and dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line.strip()) for line in f]

    print(f"Processing {len(model_paths)} models with {len(data_list)} samples")

    for model_path in tqdm(model_paths, desc="Models"):
        if num_gpus > 1:
            _process_model_multi_gpu(model_path, data_list, dataset_path, output_dir, batch_size, num_gpus)
        else:
            _process_model_single_gpu(model_path, data_list, dataset_path, output_dir, batch_size)

    return output_dir


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _mix_datasets(all_datasets: Dict, mix_strategy: str, max_samples: int = None) -> List[Dict]:
    """Mix datasets according to strategy"""
    mixed_data = []

    if mix_strategy == "balanced":
        num_datasets = len(all_datasets)
        samples_per_dataset = (max_samples // num_datasets) if max_samples else min(len(data) for data in all_datasets.values())

        for task, data in all_datasets.items():
            actual_samples = min(samples_per_dataset, len(data))
            sampled_data = random.sample(data, actual_samples)
            for hidden_states, score in sampled_data:
                mixed_data.append({"hidden_states": hidden_states, "acc_label": score, "task": task})

    elif mix_strategy == "proportional":
        total_available = sum(len(data) for data in all_datasets.values())
        target_size = min(max_samples or 10000, total_available)

        for task, data in all_datasets.items():
            proportion = len(data) / total_available
            task_samples = min(int(target_size * proportion), len(data))
            if task_samples > 0:
                sampled_data = random.sample(data, task_samples)
                for hidden_states, score in sampled_data:
                    mixed_data.append({"hidden_states": hidden_states, "acc_label": score, "task": task})

    elif mix_strategy == "all":
        all_flat = []
        for task, data in all_datasets.items():
            for hidden_states, score in data:
                all_flat.append({"hidden_states": hidden_states, "acc_label": score, "task": task})

        if max_samples and len(all_flat) > max_samples:
            mixed_data = random.sample(all_flat, max_samples)
        else:
            mixed_data = all_flat

    random.shuffle(mixed_data)
    return mixed_data


def generate_logits(config: PipelineConfig, task: str, task_path: str) -> Path:
    """Generate logits (and hidden states) for a task. Returns the logits file path."""
    print(f"🚀 Starting complete probe training pipeline for task: {task}")

    dataset_path = task_path
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} not found. Please run get_score first.")

    logits_output_dir = Path(config.training.logits_output_dir or "logits_output")
    weak_model_name = Path(config.inference.weak_model_path).name
    # Support mmlu_pro subdirectory for logits
    if task.startswith("mmlu_pro_"):
        weak_logits_file = logits_output_dir / "mmlu_pro" / f"{weak_model_name}_{task}.pt"
    else:
        weak_logits_file = logits_output_dir / f"{weak_model_name}_{task}.pt"

    if weak_logits_file.exists():
        print(f"🔄 Found existing logits files: {weak_logits_file}")
        print("⏭️  Skipping logits generation step...")
    else:
        print("📊 Step 1: Generating logits and hidden states")
        generate_logits_for_models(
            model_paths=[config.inference.weak_model_path],
            dataset_path=dataset_path,
            output_dir=str(logits_output_dir),
            batch_size=16,
            num_gpus=4
        )

    return weak_logits_file


def complete_probe_training_pipeline_with_mixed_datasets(
    config: PipelineConfig,
    task_list: list[str],
    mix_strategy: str = "balanced",
    max_samples: int = None,
    save_subdir: Optional[str] = None,
    custom_save_name: Optional[str] = None,
):
    """Train probe model using multiple mixed datasets"""
    print(f"🚀 Starting mixed dataset probe training for tasks: {task_list}")
    print(f"📊 Mix strategy: {mix_strategy}")
    if max_samples:
        print(f"🎯 Max samples: {max_samples}")

    logits_output_dir = Path(config.training.logits_output_dir or "logits_output")
    weak_model_name = Path(config.inference.weak_model_path).name

    # Load all datasets
    all_datasets = {}
    dataset_stats = {}

    for task in task_list:
        # Support mmlu_pro subdirectory for logits
        if task.startswith("mmlu_pro_"):
            pt_file = logits_output_dir / "mmlu_pro" / f"{weak_model_name}_{task}.pt"
        else:
            pt_file = logits_output_dir / f"{weak_model_name}_{task}.pt"
        if not pt_file.exists():
            print(f"⚠️  Warning: Logits file {pt_file} not found, skipping task: {task}")
            continue

        print(f"📁 Loading dataset for task: {task}")
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        if not data:
            print(f"⚠️  Warning: Empty dataset for task: {task}, skipping...")
            continue

        all_datasets[task] = data
        positive_count = sum(1 for _, score in data if score > 0.5)
        dataset_stats[task] = {
            "total": len(data), "positive": positive_count, "negative": len(data) - positive_count,
            "positive_ratio": positive_count / len(data) if len(data) > 0 else 0
        }
        print(f"   📊 {task}: {len(data)} samples ({positive_count} pos, {len(data) - positive_count} neg)")

    if not all_datasets:
        raise ValueError("No valid datasets found!")

    # Mix datasets
    print(f"🔄 Mixing datasets using strategy: {mix_strategy}")
    mixed_training_data = _mix_datasets(all_datasets, mix_strategy, max_samples)

    # Statistics
    total_samples = len(mixed_training_data)
    total_positive = sum(1 for sample in mixed_training_data if sample["acc_label"] > 0.5)
    task_counts = {}
    for sample in mixed_training_data:
        task_counts[sample["task"]] = task_counts.get(sample["task"], 0) + 1

    print(f"Mixed dataset: {total_samples} total samples")
    print(f"   Positive samples: {total_positive} ({total_positive/total_samples*100:.1f}%)")

    # Train/Val split
    split_idx = int(len(mixed_training_data) * 0.8)
    train_data = mixed_training_data[:split_idx]
    val_data = mixed_training_data[split_idx:]

    print(f"Train/Val split: {len(train_data)} train, {len(val_data)} val samples")

    
    # Train probe
    print("Training probe model on mixed datasets")
    probe_type = config.router.probe_type
    epochs = config.training.epochs
    batch_size = config.training.batch_size
    lr = config.training.learning_rate

    # Extract probe configuration from training config
    probe_config = {
        "mlp_hidden_dims": config.training.mlp_hidden_dims,
        "mlp_dropout": config.training.mlp_dropout,
        "conv_channels": config.training.conv_channels,
        "conv_kernel_size": config.training.conv_kernel_size,
    }

    # Decide save directory and filename
    save_dir = os.path.join(config.training.probe_save_path, save_subdir) if save_subdir else config.training.probe_save_path
    os.makedirs(save_dir, exist_ok=True)

    if custom_save_name:
        filename = custom_save_name if custom_save_name.endswith('.pt') else f"{custom_save_name}.pt"
    else:
        task_suffix = "_".join(sorted(task_list))
        sample_suffix = f"{max_samples}samples" if max_samples else ""
        filename = f"{sample_suffix}_mixed_{task_suffix}_{probe_type}.pt"

    save_path = os.path.join(save_dir, filename)

    results = train_probe_model(train_data, val_data, probe_type, save_path,
                               probe_config=probe_config, epochs=epochs, batch_size=batch_size, lr=lr)

    print(f"Mixed dataset probe training complete!")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Model saved to: {save_path}")

    return {
        "model_path": save_path, "training_results": results, "dataset_stats": dataset_stats,
        "mixed_stats": {"total_samples": total_samples, "positive_samples": total_positive,
                       "task_distribution": task_counts, "mix_strategy": mix_strategy, "max_samples": max_samples},
        "train_samples": len(train_data), "val_samples": len(val_data), "tasks_used": list(all_datasets.keys())
    }


def complete_layerwise_probe_training_pipeline(config: PipelineConfig, task_list: list[str],
                                             mix_strategy: str = "balanced", max_samples: int = None):
    """Train probe models for each layer using hs_mlp automatically"""
    print(f"🚀 Starting layer-wise probe training for tasks: {task_list}")
    print(f"📊 Mix strategy: {mix_strategy}")
    if max_samples:
        print(f"🎯 Max samples: {max_samples}")

    logits_output_dir = Path(config.training.logits_output_dir or "logits_output")
    weak_model_name = Path(config.inference.weak_model_path).name

    # Load and mix datasets (reuse logic)
    all_datasets = {}
    dataset_stats = {}

    for task in task_list:
        
        pt_file = logits_output_dir / f"{weak_model_name}_{task}.pt"
        if not pt_file.exists():
            print(f"⚠️ Warning: Logits file {pt_file} not found, skipping task: {task}")
            continue

        print(f"📁 Loading dataset for task: {task}")
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        if not data:
            print(f"⚠️ Warning: Empty dataset for task: {task}, skipping...")
            continue

        all_datasets[task] = data
        positive_count = sum(1 for _, score in data if score > 0.5)
        dataset_stats[task] = {
            "total": len(data), "positive": positive_count, "negative": len(data) - positive_count,
            "positive_ratio": positive_count / len(data) if len(data) > 0 else 0
        }
        print(f"   📊 {task}: {len(data)} samples ({positive_count} pos, {len(data) - positive_count} neg)")

    if not all_datasets:
        raise ValueError("No valid datasets found!")

    # Mix datasets
    print(f"🔄 Mixing datasets using strategy: {mix_strategy}")
    mixed_training_data = _mix_datasets(all_datasets, mix_strategy, max_samples)

    total_samples = len(mixed_training_data)
    print(f"📊 Mixed dataset: {total_samples} total samples")

    # Train/Val split
    split_idx = int(len(mixed_training_data) * 0.8)
    train_data = mixed_training_data[:split_idx]
    val_data = mixed_training_data[split_idx:]

    print(f"📊 Train/Val split: {len(train_data)} train, {len(val_data)} val samples")

    # Detect number of layers
    sample_hidden_states = train_data[0]["hidden_states"]
    num_layers = sample_hidden_states.shape[0]
    print(f"🧠 Detected {num_layers} layers in hidden states")

    # Training setup
    epochs = config.training.epochs or 50
    batch_size = config.training.batch_size or 32
    lr = config.training.learning_rate or 1e-4

    # Extract probe configuration from training config
    probe_config = {
        "mlp_hidden_dims": config.training.mlp_hidden_dims,
        "mlp_dropout": config.training.mlp_dropout,
        "conv_channels": config.training.conv_channels,
        "conv_kernel_size": config.training.conv_kernel_size,
    }

    all_layer_results = {}
    task_suffix = "_".join(sorted(task_list))
    sample_suffix = f"_{max_samples}samples" if max_samples else ""

    print(f"🚀 Starting training for all {num_layers} layers...")

    # Train probe for each layer
    for layer_idx in range(num_layers):
        print(f"\n🧠 Training probe for layer {layer_idx}/{num_layers-1}")

        # Create layer-specific training data
        layer_train_data = []
        layer_val_data = []

        for item in train_data:
            layer_hidden_states = item["hidden_states"][layer_idx]
            if isinstance(layer_hidden_states, torch.Tensor):
                layer_hidden_states = layer_hidden_states.cpu().numpy()

            layer_train_data.append({
                "hidden_states": layer_hidden_states,
                "acc_label": item["acc_label"],
                "task": item["task"]
            })

        for item in val_data:
            layer_hidden_states = item["hidden_states"][layer_idx]
            if isinstance(layer_hidden_states, torch.Tensor):
                layer_hidden_states = layer_hidden_states.cpu().numpy()

            layer_val_data.append({
                "hidden_states": layer_hidden_states,
                "acc_label": item["acc_label"],
                "task": item["task"]
            })

        # Create save path for this layer
        layer_save_path = os.path.join(
            config.training.probe_save_path,
            f"mixed_{task_suffix}_{mix_strategy}{sample_suffix}_layer{layer_idx}_probe_hs_mlp.pt"
        )

        # Train using hs_mlp probe type
        layer_results = train_probe_model(
            layer_train_data, layer_val_data, "hs_mlp", layer_save_path,
            probe_config=probe_config, epochs=epochs, batch_size=batch_size, lr=lr
        )

        all_layer_results[f"layer_{layer_idx}"] = {
            **layer_results,
            "model_path": layer_save_path,
            "layer_idx": layer_idx
        }

        print(f"💾 Layer {layer_idx} model saved to: {layer_save_path}")

    print(f"\n✅ All layer-wise probe training complete!")

    # Summary
    print(f"\n📊 Layer-wise Performance Summary:")
    for layer_key, result in all_layer_results.items():
        layer_idx = result['layer_idx']
        best_loss = result['best_val_loss']
        print(f"   Layer {layer_idx}: Val Loss = {best_loss:.4f}")

    return {
        "layer_results": all_layer_results,
        "dataset_stats": dataset_stats,
        "mixed_stats": {"total_samples": total_samples, "mix_strategy": mix_strategy, "max_samples": max_samples},
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "tasks_used": list(all_datasets.keys()),
        "num_layers": num_layers
    }


# def train_probe_from_config(config: PipelineConfig, task: str):
#     """Train probe using config parameters and task data"""
#     return complete_probe_training_pipeline(config, task)


if __name__ == "__main__":
    if fire is None:
        raise SystemExit("Missing optional dependency 'fire'. Install with: pip install fire")
    fire.Fire()
