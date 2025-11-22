class DynamicFusionProbe(nn.Module):
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

