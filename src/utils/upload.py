"""
Upload DynamicFusionProbe model to Hugging Face
"""
import json
import os
os.environ['HF_TOKEN'] = 'hf'  # Set your token here or in environment variables
from huggingface_hub import HfApi, create_repo, login

# ============= Config =============
REPO_NAME = "dynamic-dirichlet-probe-qwen2.5-numina"
YOUR_USERNAME = "wanxwu"  
MODEL_PATH = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/qwen/numina_cot_5k_train_allsamples_dirichlet_probe.pt"  # Change this

repo_id = f"{YOUR_USERNAME}/{REPO_NAME}"

# ============= Mirror and Login =============
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

HF_TOKEN = os.environ.get('HF_TOKEN')

import torch
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Logged in\n")
else:
    print("❌ Please set: export HF_TOKEN=your_token")
    exit(1)

# ============= Load Model =============
print(f"📂 Loading model: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
state_dict = checkpoint['model_state_dict']
metadata = checkpoint.get('metadata', {})

print("Model parameters:")
for key, value in state_dict.items():
    print(f"  {key}: {value.shape}")
print()

# ============= Config =============
config = {
    'model_type': 'DynamicDirichletProbe',
    'num_layers': int(state_dict['concentration_logits'].shape[0]),
    'input_dim': int(state_dict['classifier.weight'].shape[1]),
    'output_dim': int(state_dict['classifier.weight'].shape[0]),
    'metadata': metadata
}

print("Config:", json.dumps(config, indent=2))
print()

# ============= Save Files =============
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

torch.save(state_dict, 'pytorch_model.bin')

# README
readme = f"""---
license: mit
tags:
- pytorch
- probe
- dirichlet
---

# Dynamic Dirichlet Probe

Dynamic layer fusion probe using Dirichlet distribution.

## Parameters

- **sentence-level hidden_states**: `[batch_size, {config['num_layers']}, {config['input_dim']}]` - Hidden states from each layer in Llama-3.1-8B
- **concentration_logits**: `[{config['num_layers']}]` - Per-layer concentration parameters
- **global_concentration**: `scalar` - Global concentration scale
- **classifier**: `Linear({config['input_dim']} → {config['output_dim']})`

## Input
```python
hidden_states: [batch_size, {config['num_layers']}, {config['input_dim']}]
```

## Usage
```python
import torch
import os
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(repo_id="{repo_id}", filename="pytorch_model.bin")
state_dict = torch.load(model_path)
```
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

# modeling.py
model_code = '''class DynamicFusionProbe(nn.Module):
    """动态融合每一层信号的probe"""
    def __init__(self, input_dim: int, num_layers: int, output_dim: int = 1, probe_type: str = "softmax"):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.probe_type = probe_type

        if probe_type == "softmax":
            # 原始方法：每层的权重参数，可学习
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        elif probe_type == "dirichlet":
            # Dirichlet方法：学习浓度参数
            self.concentration_logits = nn.Parameter(torch.ones(num_layers))  # 学习log(α)
            self.global_concentration = nn.Parameter(torch.tensor(1.0))  # 学习β₀
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        # 最终的分类器
        self.classifier = nn.Linear(input_dim, output_dim)

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
            # 计算浓度参数: α = β₀ * softmax(concentration_logits)
            base_concentration = torch.softmax(self.concentration_logits, dim=0)  # [num_layers]
            concentration = torch.exp(self.global_concentration) * base_concentration  # [num_layers]

            if self.training:
                # 训练时：从Dirichlet分布采样
                dirichlet_dist = Dirichlet(concentration)
                weights = dirichlet_dist.rsample((batch_size,))  # [batch_size, num_layers]
                weights = weights.unsqueeze(-1)  # [batch_size, num_layers, 1]

                # 计算不确定性：使用熵
                uncertainty = dirichlet_dist.entropy()  # [batch_size]
            else:
                # 推理时：使用期望值
                weights = (concentration / concentration.sum()).unsqueeze(0).unsqueeze(-1)  # [1, num_layers, 1]
                weights = weights.expand(batch_size, -1, -1)  # [batch_size, num_layers, 1]

                # 计算不确定性：基于浓度参数的总和
                total_concentration = concentration.sum()
                uncertainty = torch.log(total_concentration).expand(batch_size)

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

'''

with open('modeling.py', 'w') as f:
    f.write(model_code)

print("✅ All files saved\n")

# ============= Upload =============
print(f"🚀 Uploading to: {repo_id}\n")

try:
    create_repo(repo_id, exist_ok=True)
    print("✅ Repository created\n")
except Exception as e:
    print(f"❌ {e}\n")
    exit(1)

api = HfApi()
files = ['pytorch_model.bin', 'config.json', 'README.md', 'modeling.py']

for filename in files:
    try:
        api.upload_file(path_or_fileobj=filename, path_in_repo=filename, repo_id=repo_id)
        print(f"✅ {filename}")
    except Exception as e:
        print(f"❌ {filename}: {e}")

print(f"\n🎉 Done!")
print(f"https://huggingface.co/{repo_id}")
print(f"https://hf-mirror.com/{repo_id}")