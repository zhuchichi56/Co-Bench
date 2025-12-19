import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Dirichlet
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
import random
from tqdm import tqdm
import os

class DynamicFusionProbe(nn.Module):
    """动态融合每一层信号的probe"""
    def __init__(self, input_dim: int, num_layers: int, output_dim: int = 1, probe_type: str = "softmax",
                 mlp_hidden_dims: List[int] = None, dropout: float = 0.1, use_input_dependent: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.probe_type = probe_type
        self.use_input_dependent = use_input_dependent

        if probe_type == "softmax":
            # 原始方法：每层的权重参数，可学习
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        elif probe_type == "dirichlet":
            if use_input_dependent:
                # 输入依赖版本：通过网络计算 β(x)
                hidden_dim = max(num_layers * 2, 64)
                self.concentration_network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_layers),
                    nn.Softplus()  # 确保输出为正数
                )
            else:
                # 原版本：固定的全局浓度参数
                self.concentration_logits = nn.Parameter(torch.ones(num_layers))  # 学习log(α)
                self.global_concentration = nn.Parameter(torch.tensor(1.0))  # 学习β₀
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        # 最终的分类器（支持多层 MLP）
        if mlp_hidden_dims is None or len(mlp_hidden_dims) == 0:
            # 单层分类器
            self.classifier = nn.Linear(input_dim, output_dim)
        else:
            # 多层 MLP
            layers = []
            prev_dim = input_dim
            for hidden_dim in mlp_hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.classifier = nn.Sequential(*layers)

    def forward(self, hidden_states, return_uncertainty=False):
        """
        Args:
            hidden_states: [batch_size, num_layers, hidden_dim]
            return_uncertainty: 是否返回不确定性指标 (仅对Dirichlet有效)
        Returns:
            logits: [batch_size, output_dim]
            uncertainty: (optional) 不确定性指标
        """
        batch_size = hidden_states.size(0)

        if self.probe_type == "softmax":
            # 原始方法：简单softmax权重
            weights = torch.softmax(self.layer_weights, dim=0)  # [num_layers]
            weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, num_layers, 1]
            fused_features = torch.sum(hidden_states * weights, dim=1)  # [batch_size, hidden_dim]

            logits = self.classifier(fused_features)

            if return_uncertainty:
                return logits, None  # 原始方法不提供不确定性
            return logits

        elif self.probe_type == "dirichlet":
            # Dirichlet方法：从Dirichlet分布采样权重
            if self.use_input_dependent:
                # 输入依赖版本：β(x) 根据输入计算
                input_features = hidden_states.mean(dim=1)  # [batch_size, input_dim]
                concentration = self.concentration_network(input_features)  # [batch_size, num_layers]
                concentration = concentration + 1e-6  # 数值稳定性
            else:
                # 原版本：固定的全局浓度参数
                base_concentration = torch.softmax(self.concentration_logits, dim=0)  # [num_layers]
                concentration = torch.exp(self.global_concentration) * base_concentration  # [num_layers]
                # 扩展到 batch 维度
                concentration = concentration.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_layers]

            if self.training:
                # 训练时：从Dirichlet分布采样
                weights_list = []
                uncertainty_list = []
                for i in range(batch_size):
                    dirichlet_dist = Dirichlet(concentration[i])
                    sample_weights = dirichlet_dist.rsample()  # [num_layers]
                    weights_list.append(sample_weights)
                    uncertainty_list.append(dirichlet_dist.entropy())

                weights = torch.stack(weights_list, dim=0)  # [batch_size, num_layers]
                uncertainty = torch.stack(uncertainty_list, dim=0)  # [batch_size]
                weights = weights.unsqueeze(-1)  # [batch_size, num_layers, 1]
            else:
                # 推理时：使用期望值 ᾱ_l(x) = β_l(x) / Σβ_j(x)
                beta_0 = concentration.sum(dim=1, keepdim=True)  # [batch_size, 1]
                weights = concentration / beta_0  # [batch_size, num_layers]
                weights = weights.unsqueeze(-1)  # [batch_size, num_layers, 1]

                # 不确定性：使用 -log(β₀(x))
                uncertainty = -torch.log(beta_0.squeeze(1))  # [batch_size]

            # 加权融合
            fused_features = torch.sum(hidden_states * weights, dim=1)  # [batch_size, hidden_dim]
            logits = self.classifier(fused_features)

            if return_uncertainty:
                return logits, uncertainty
            return logits


class DynamicProbeDataset(Dataset):
    def __init__(self, data: List[Tuple[np.ndarray, float]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hidden_states, label = self.data[idx]
        hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return hidden_states, label


def train_dynamic_probe(train_data: List[Tuple[np.ndarray, float]],
                       val_data: List[Tuple[np.ndarray, float]],
                       epochs: int = 50,
                       batch_size: int = 32,
                       lr: float = 1e-4,
                       save_path: str = None,
                       probe_type: str = "softmax",
                       mlp_hidden_dims: List[int] = None,
                       dropout: float = 0.1,
                       use_input_dependent: bool = False) -> Dict:
    """训练动态融合probe"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建数据集
    train_dataset = DynamicProbeDataset(train_data)
    val_dataset = DynamicProbeDataset(val_data)

    # 获取输入维度和层数
    sample_hidden_states, _ = train_data[0]
    num_layers, input_dim = sample_hidden_states.shape

    print(f"Input dim: {input_dim}, Num layers: {num_layers}")

    # 创建模型
    model = DynamicFusionProbe(input_dim, num_layers, probe_type=probe_type,
                              mlp_hidden_dims=mlp_hidden_dims, dropout=dropout,
                              use_input_dependent=use_input_dependent).to(device)
    print(f"Using probe type: {probe_type}")
    print(f"Input dependent: {use_input_dependent}")
    if mlp_hidden_dims:
        print(f"MLP hidden dims: {mlp_hidden_dims}, dropout: {dropout}")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    val_accuracies = []
    val_aurocs = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features).squeeze(-1)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features).squeeze(-1)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predictions = probs > 0.5
                correct += (predictions == batch_labels.bool()).sum().item()
                total += batch_labels.size(0)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total

        # 计算 AUROC
        try:
            from sklearn.metrics import roc_auc_score
            val_auroc = roc_auc_score(all_labels, all_probs)
        except:
            val_auroc = 0.0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_aurocs.append(val_auroc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}, Val AUROC={val_auroc:.4f}")

        # 打印学习到的层权重/浓度参数
        if epoch % 10 == 0:
            if probe_type == "softmax":
                weights = torch.softmax(model.layer_weights, dim=0)
                print(f"Layer weights: {weights.detach().cpu().numpy()}")
            elif probe_type == "dirichlet" and use_input_dependent == False :
                base_concentration = torch.softmax(model.concentration_logits, dim=0)
                concentration = torch.exp(model.global_concentration) * base_concentration
                print(f"Concentration params: {concentration.detach().cpu().numpy()}")
                print(f"Global concentration (β₀): {torch.exp(model.global_concentration).item():.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metadata': {
                        'input_dim': input_dim,
                        'num_layers': num_layers,
                        'output_dim': 1,
                        'probe_type': probe_type,
                        'mlp_hidden_dims': mlp_hidden_dims,
                        'dropout': dropout,
                        'use_input_dependent': use_input_dependent
                    }
                }, save_path)
                print(f"Best model saved to {save_path}")

    # 返回最终权重/浓度参数
    if probe_type == "softmax":
        final_weights = torch.softmax(model.layer_weights, dim=0).detach().cpu().numpy()
        extra_info = {'final_layer_weights': final_weights}
    elif probe_type == "dirichlet":
        base_concentration = torch.softmax(model.concentration_logits, dim=0)
        concentration = torch.exp(model.global_concentration) * base_concentration
        final_weights = (concentration / concentration.sum()).detach().cpu().numpy()
        extra_info = {
            'final_layer_weights': final_weights,
            'final_concentration': concentration.detach().cpu().numpy(),
            'final_global_concentration': torch.exp(model.global_concentration).item()
        }

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_aurocs': val_aurocs,
        'best_val_loss': best_val_loss,
        'probe_type': probe_type,
        **extra_info
    }


def test_dynamic_probe(test_data: List[Tuple[np.ndarray, float]],
                      model_path: str) -> Dict:
    """测试动态融合probe"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint['metadata']

    probe_type = metadata.get('probe_type', 'softmax')  # 向后兼容
    use_input_dependent = metadata.get('use_input_dependent', False)  # 向后兼容
    model = DynamicFusionProbe(
        metadata['input_dim'],
        metadata['num_layers'],
        metadata['output_dim'],
        probe_type=probe_type,
        mlp_hidden_dims=metadata.get('mlp_hidden_dims', None),
        dropout=metadata.get('dropout', 0.1),
        use_input_dependent=use_input_dependent
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建测试数据集
    test_dataset = DynamicProbeDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_features = batch_features.to(device)

            outputs = model(batch_features).squeeze(-1)
            predictions = torch.sigmoid(outputs).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(batch_labels.numpy())

    # 计算性能指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    binary_predictions = (all_predictions > 0.5).astype(int)
    accuracy = (binary_predictions == all_labels).mean()

    # 打印最终的层权重/浓度参数
    if probe_type == "softmax":
        weights = torch.softmax(model.layer_weights, dim=0).detach().cpu().numpy()
        print(f"Final layer weights: {weights}")
        extra_info = {'layer_weights': weights}
    elif probe_type == "dirichlet":
        base_concentration = torch.softmax(model.concentration_logits, dim=0)
        concentration = torch.exp(model.global_concentration) * base_concentration
        weights = (concentration / concentration.sum()).detach().cpu().numpy()
        print(f"Final layer weights: {weights}")
        print(f"Final concentration: {concentration.detach().cpu().numpy()}")
        print(f"Global concentration (β₀): {torch.exp(model.global_concentration).item():.4f}")
        extra_info = {
            'layer_weights': weights,
            'concentration': concentration.detach().cpu().numpy(),
            'global_concentration': torch.exp(model.global_concentration).item()
        }

    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'probe_type': probe_type,
        **extra_info
    }


def run_dynamic_probe_pipeline(task: str,
                              hidden_states_file: str,
                              save_dir: str = "probe_save",
                              probe_type: str = "softmax",
                              mlp_hidden_dims: List[int] = None,
                              dropout: float = 0.1,
                              use_input_dependent: bool = False):
    """运行完整的动态probe训练和测试流程"""

    print(f"Running dynamic probe pipeline for task: {task}")

    # 加载数据
    print(f"Loading data from {hidden_states_file}")
    data = torch.load(hidden_states_file, map_location="cpu",weights_only= False)

    if not data:
        raise ValueError(f"No data found in {hidden_states_file}")

    print(f"Loaded {len(data)} samples")

    # 统计标签分布
    positive_count = sum(1 for _, score in data if score > 0.5)
    negative_count = len(data) - positive_count
    print(f"Label distribution: {positive_count} positive, {negative_count} negative")

    # 数据分割
    random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task}_{probe_type}_probe.pt")

    # 训练
    print(f"Training dynamic fusion probe with {probe_type} method...")
    results = train_dynamic_probe(train_data, val_data, save_path=save_path, probe_type=probe_type,
                                 mlp_hidden_dims=mlp_hidden_dims, dropout=dropout,
                                 use_input_dependent=use_input_dependent)

    print(f"Training completed. Best val loss: {results['best_val_loss']:.4f}")
    print(f"Final layer weights: {results['final_layer_weights']}")

    # 测试
    print("Testing dynamic fusion probe...")
    test_results = test_dynamic_probe(val_data, save_path)

    print(f"Test accuracy: {test_results['accuracy']:.4f}")

    # 保存训练历史和配置到 JSON 文件
    history_file = os.path.join(save_dir, f"{task}_{probe_type}_history.json")
    history_data = {
        'config': {
            'task': task,
            'probe_type': probe_type,
            'mlp_hidden_dims': mlp_hidden_dims,
            'dropout': dropout,
            'use_input_dependent': use_input_dependent,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        },
        'training_history': {
            'train_losses': results['train_losses'],
            'val_losses': results['val_losses'],
            'val_accuracies': results['val_accuracies'],
            'val_aurocs': results['val_aurocs']
        },
        'final_results': {
            'best_val_loss': float(results['best_val_loss']),
            'test_accuracy': float(test_results['accuracy']),
            'final_layer_weights': results['final_layer_weights'].tolist() if hasattr(results['final_layer_weights'], 'tolist') else list(results['final_layer_weights'])
        }
    }

    # 如果是 dirichlet 类型，添加额外的信息
    if probe_type == "dirichlet":
        history_data['final_results']['final_concentration'] = results['final_concentration'].tolist() if hasattr(results['final_concentration'], 'tolist') else list(results['final_concentration'])
        history_data['final_results']['final_global_concentration'] = float(results['final_global_concentration'])

    import json
    with open(history_file, 'w') as f:
        json.dump(history_data, f, indent=2)

    print(f"📊 Training history saved to: {history_file}")

    return {
        'training_results': results,
        'test_results': test_results,
        'model_path': save_path,
        'history_file': history_file
    }


if __name__ == "__main__":
    # 示例使用
    task = "math"
    hidden_states_file = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/logits/mmlu_pro/Qwen2.5-7B-Instruct_math.pt"

    if os.path.exists(hidden_states_file):
        results = run_dynamic_probe_pipeline(task, hidden_states_file)
        print("Pipeline completed successfully!")
    else:
        print(f"Hidden states file not found: {hidden_states_file}")