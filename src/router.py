import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModel
from torch.distributions import Dirichlet
from inference.vllm_client import parallel_inference
from scipy.stats import entropy as scipy_entropy


class Router(ABC):
    @abstractmethod
    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        pass


# finish 
class SelfQuestioningRouter(Router):
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

    def parse_boxed_number(self, text: str) -> Optional[float]:
        import re
        # Try to find \boxed{number}
        match = re.search(r'\\boxed\{([0-9]+(?:\.[0-9]*)?)\}', text)
        if match:
            return float(match.group(1))
        # Try to find [number]
        match = re.search(r'\[([0-9]+(?:\.[0-9]*)?)\]', text)
        if match:
            return float(match.group(1))
        # Try to find just a number
        match = re.search(r'([0-9]+(?:\.[0-9]*)?)', text)
        if match:
            return float(match.group(1))
        return None

    def get_router_scores(self, data: List[Dict], model_type: str = "weak", **kwargs) -> np.ndarray:
        prompts = []
        for item in data:
            question = item.get("instruction", "")

            # Only use instruction, no answer - router should judge difficulty without seeing the answer
            confidence_prompt = f"""Question: {question}

On a scale of 0-100, how easy is this question for you to answer correctly? Rate 0 for very difficult, 100 for very easy. Please write your answer as \\boxed{{number}}."""

            prompts.append(confidence_prompt)

        confidence_responses = parallel_inference(
            prompts,
            max_tokens=10,
            temperature=0.0,
            model_name_or_path=self.model_path,
            type=model_type
        )

        scores = []
        for response in confidence_responses:
            try:
                confidence = self.parse_boxed_number(response)
                if confidence is not None:
                    confidence = max(0, min(100, confidence)) / 100.0
                    scores.append(confidence)
                else:
                    scores.append(0.5)
            except Exception:
                scores.append(0.5)

        return np.array(scores)


class DebertaRouter(Router):
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1).to(self.device)

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        self.model.eval()
        scores = []

        with torch.no_grad():
            for item in data:
                question = item.get("instruction", "")
                response = item.get("generated_response", "")

                text = f"Question: {question} Answer: {response}"

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.model(**inputs)
                pooled_output = outputs.last_hidden_state.mean(dim=1)
                logit = self.classifier(pooled_output)
                score = torch.sigmoid(logit).cpu().item()
                scores.append(score)

        return np.array(scores)


class TrainedDebertaRouter(Router):
    """Router using trained DeBERTa model for question classification"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained model and tokenizer
        try:
            from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
            self.model = DebertaV2ForSequenceClassification.from_pretrained(model_path).to(self.device)
        except:
            # Fallback to AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)

        self.sep_token = "<SEP>"

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        self.model.eval()
        scores = []

        with torch.no_grad():
            for item in data:
                question = item.get("instruction", "")
                llm_id = item.get("llm_id", "unknown")

                # Format as: Question <SEP> llm_id <SEP>
                text = f"{question} {self.sep_token} {llm_id} {self.sep_token}"

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.model(**inputs)

                # If it's a classification model, get logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    if logits.size(-1) == 1:
                        score = torch.sigmoid(logits).cpu().item()
                    else:
                        # Multi-class, take probability of class 1
                        probs = torch.softmax(logits, dim=-1)
                        score = probs[0, 1].cpu().item() if logits.size(-1) > 1 else probs[0, 0].cpu().item()
                else:
                    # Fallback: use mean pooling
                    pooled_output = outputs.last_hidden_state.mean(dim=1)
                    score = torch.sigmoid(pooled_output.mean()).cpu().item()

                scores.append(score)

        return np.array(scores)


class LLMRouter(Router):
    """Router using trained LLM for question difficulty assessment"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained LLM model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(self.device)

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        self.model.eval()
        scores = []

        # Prompt template for difficulty assessment
        prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
On a scale of 0.0 to 1.0, rate the difficulty of this question where 0.0 means very easy and 1.0 means very difficult: {question}

### Response:"""

        with torch.no_grad():
            for item in data:
                question = item.get("instruction", "")

                # Format prompt
                prompt = prompt_template.format(question=question)

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # Decode and parse response
                generated_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

                # Parse difficulty score
                try:
                    import re
                    # Look for decimal numbers between 0 and 1
                    matches = re.findall(r'0\.\d+|1\.0|0|1', generated_text)
                    if matches:
                        score = float(matches[0])
                        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                    else:
                        score = 0.5  # Default if parsing fails
                except:
                    score = 0.5

                scores.append(score)

        return np.array(scores)


class ZScoreNormalizer:
    def __init__(self, mu: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mu = mu.float().cpu()
        self.std = std.float().clamp_min(eps).cpu()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float().cpu() - self.mu) / self.std


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


class ConvProbe(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, conv_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=conv_channels,
                              kernel_size=kernel_size, padding=pad)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_channels, out_dim)

    def forward(self, x):
        h = self.conv(x)
        h = F.relu(h)
        h = self.gap(h).squeeze(-1)
        return self.fc(h)


class MeanProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x.mean(dim=1))


class MaxProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x.max(dim=1).values)


class MeanMaxProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x):
        # mean_feat = x.mean(dim=-1)
        # max_feat = x.max(dim=-1)[0]

        mean_across_layers = x.mean(dim=1)  
        max_across_layers = x.max(dim=1).values   
        combined = torch.cat([mean_across_layers, max_across_layers], dim=-1)
  
        return self.fc(combined)


class TransformerProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))


class DynamicFusionProbe(nn.Module):
    """动态融合每一层信号的probe，支持softmax和Dirichlet两种方法"""
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



class DynamicFusionRouter(Router):
    """基于动态融合probe的Router"""

    def __init__(self, checkpoint_path: str, probe_type: str = "softmax", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.probe_type = probe_type
        self.model, self.metadata = self.load_dynamic_fusion_probe(checkpoint_path)
        self.model.to(self.device)

    def load_dynamic_fusion_probe(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model_state = checkpoint["model_state_dict"]
        metadata = checkpoint.get("metadata", {})

        input_dim = metadata.get("input_dim", 4096)
        num_layers = metadata.get("num_layers", 32)
        output_dim = metadata.get("output_dim", 1)
        probe_type = metadata.get("probe_type", self.probe_type)

        model = DynamicFusionProbe(input_dim, num_layers, output_dim, probe_type)
        # Compatibility: remap checkpoint keys between fc.* and net.* when needed
        try:
            expected_keys = set(model.state_dict().keys())
            ckpt_keys = set(model_state.keys())

            # Case 1: model expects net.* but checkpoint provides fc.*
            if any(k.startswith("net.") for k in expected_keys) and any(k.startswith("fc.") for k in ckpt_keys):
                remapped = {}
                for k, v in model_state.items():
                    new_k = k.replace("fc.", "net.") if k.startswith("fc.") else k
                    remapped[new_k] = v
                model_state = remapped

            # Case 2: model expects fc.* but checkpoint provides net.*
            if any(k.startswith("fc.") for k in expected_keys) and any(k.startswith("net.") for k in ckpt_keys):
                remapped = {}
                for k, v in model_state.items():
                    new_k = k.replace("net.", "fc.") if k.startswith("net.") else k
                    remapped[new_k] = v
                model_state = remapped

            missing, unexpected = [], []
            try:
                # dry run to collect issues without throwing
                model.load_state_dict(model_state, strict=False)
                # When strict=False, we cannot directly get missing/unexpected; do a manual diff for logging
                missing = [k for k in expected_keys if k not in model_state]
                unexpected = [k for k in model_state if k not in expected_keys]
                if missing or unexpected:
                    print(f"[Probe Load] Non-strict load. Missing: {missing}, Unexpected: {unexpected}")
            except Exception:
                # fallback to strict load to expose error
                model.load_state_dict(model_state)
        except Exception:
            # As a last resort, try original loading
            model.load_state_dict(model_state)

        return model, metadata

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        self.model.eval()

        features = []
        for item in data:
            # 处理元组格式的数据
            if isinstance(item, tuple):
                hidden_states = item[0]  # numpy.ndarray
            else:
                # 处理字典格式的数据
                hidden_states = item.get("hidden_states", [])

            # 将 numpy 数组转换为 torch tensor
            if isinstance(hidden_states, np.ndarray):
                hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
            elif not isinstance(hidden_states, torch.Tensor):
                hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

            features.append(hidden_states.unsqueeze(0))

        features = torch.cat(features, dim=0).to(self.device)

        with torch.no_grad():
            logits = self.model(features)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()

        return scores

class ProbeRouter(Router):
    PROBE_TYPES = {
        "hs_last_mlp": MLPProbe,
        "hs_mlp":MLPProbe,
        "coe_dual_mlp": MLPProbe,
        "coe_c_scalar": MLPProbe,
        "coe_r_scalar": MLPProbe,
        "pca_conv": ConvProbe,
        "mean": MeanProbe,
        "max": MaxProbe,
        "mean+max": MeanMaxProbe,
        "transformer": TransformerProbe,
        "dynamic_softmax": DynamicFusionProbe,
        "dynamic_dirichlet": DynamicFusionProbe
    }

    def __init__(self, checkpoint_path: str, probe_type: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.probe_type = probe_type
        self.model, self.normalizer, self.metadata = self.load_probe_from_checkpoint(checkpoint_path)
        self.model.to(self.device)

    def load_probe_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model_state = checkpoint["model_state_dict"]
        metadata = checkpoint.get("metadata", {})
        normalizer_state = checkpoint.get("normalizer", None)
             
        input_dim = metadata.get("input_dim", 4096)
        output_dim = metadata.get("output_dim", 1)

        if self.probe_type not in self.PROBE_TYPES:
            raise ValueError(f"Unknown probe type: {self.probe_type}")

        model_class = self.PROBE_TYPES[self.probe_type]

        if self.probe_type in ["dynamic_softmax", "dynamic_dirichlet"]:
            # 动态融合probe需要额外的参数
            num_layers = metadata.get("num_layers", 32)
            probe_method = "softmax" if self.probe_type == "dynamic_softmax" else "dirichlet"
            model = model_class(input_dim, num_layers, output_dim, probe_method)
        elif self.probe_type == "pca_conv":
            model = model_class(input_dim, output_dim)
        elif self.probe_type == "transformer":
            model = model_class(input_dim, output_dim)
        else:
            model = model_class(input_dim, output_dim)

        model.load_state_dict(model_state)

        normalizer = None
        if normalizer_state:
            normalizer = ZScoreNormalizer(
                normalizer_state["mu"],
                normalizer_state["std"]
            )

        return model, normalizer, metadata

    def extract_features(self, data: List[Dict]) -> torch.Tensor:
        features = []
        for i, item in enumerate(data):
            # 处理元组格式的数据
            if isinstance(item, tuple):
                # 假设第一个元素是 hidden_states，第二个是标签
                hidden_states = item[0]  # numpy.ndarray
                # 如果需要标签，可以用 item[1]
            else:
                # 处理字典格式的数据
                hidden_states = item.get("hidden_states", [])
            
            # 将 numpy 数组转换为 torch tensor
            if isinstance(hidden_states, np.ndarray):
                hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
            elif not isinstance(hidden_states, torch.Tensor):
                hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

            if self.probe_type == "hs_last_mlp" or self.probe_type =="hs_mlp":
                feat = hidden_states[-1]
            elif self.probe_type in ["coe_dual_mlp", "coe_c_scalar", "coe_r_scalar"]:
                # 加入计算 coe 的代码
                mag_features = []
                angle_features = []
                for j in range(hidden_states.shape[0] - 1):
                    h_curr = hidden_states[j]
                    h_next = hidden_states[j+1]

                    mag = torch.norm(h_curr, p=2) + torch.norm(h_next, p=2)
                    mag_features.append(mag)

                    cos_sim = F.cosine_similarity(h_curr, h_next, dim=0)
                    angle_features.append(cos_sim)

                feat = torch.cat([
                    torch.stack(mag_features),
                    torch.stack(angle_features)
                ], dim=0)
            else:
                feat = hidden_states  # 修正：应该是 feat 而不是 features

            features.append(feat.unsqueeze(0))

        return torch.cat(features, dim=0)

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        self.model.eval()
        features = self.extract_features(data)

        if self.normalizer:
            features = self.normalizer(features)

        with torch.no_grad():
            features = features.to(self.device)

            if self.probe_type == "pca_conv":
                features = features.permute(0, 2, 1).contiguous()

            logits = self.model(features)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)

            scores = torch.sigmoid(logits).cpu().numpy()

        return scores


class RouterManager:
    def __init__(self):
        self.routers = {}

    def register_router(self, name: str, router: Router):
        self.routers[name] = router

    def get_router_scores(self, router_name: str, data: List[Dict], **kwargs) -> np.ndarray:
        if router_name not in self.routers:
            raise ValueError(f"Unknown router: {router_name}")

        return self.routers[router_name].get_router_scores(data, **kwargs)

    def create_probe_router(self, checkpoint_path: str, probe_type: str, name: Optional[str] = None):
        router = ProbeRouter(checkpoint_path, probe_type)
        router_name = name or f"probe_{probe_type}"
        self.register_router(router_name, router)
        return router_name

    def create_self_questioning_router(self, model_path: str, name: Optional[str] = None):
        router = SelfQuestioningRouter(model_path)
        router_name = name or "self_questioning"
        self.register_router(router_name, router)
        return router_name

    def create_deberta_router(self, model_path: str, name: Optional[str] = None):
        router = DebertaRouter(model_path)
        router_name = name or "deberta"
        self.register_router(router_name, router)
        return router_name

    def create_trained_deberta_router(self, model_path: str, name: Optional[str] = None):
        router = TrainedDebertaRouter(model_path)
        router_name = name or "trained_deberta"
        self.register_router(router_name, router)
        return router_name

    def create_llm_router(self, model_path: str, name: Optional[str] = None):
        router = LLMRouter(model_path)
        router_name = name or "llm"
        self.register_router(router_name, router)
        return router_name

    def create_dynamic_fusion_router(self, checkpoint_path: str, probe_type: str = "softmax", name: Optional[str] = None):
        """创建动态融合router
        Args:
            checkpoint_path: 模型检查点路径
            probe_type: "softmax" 或 "dirichlet"
            name: router名称
        """
        router = DynamicFusionRouter(checkpoint_path, probe_type)
        router_name = name or f"dynamic_fusion_{probe_type}"
        self.register_router(router_name, router)
        return router_name

    def create_logits_margin_router(self, model_path: str, name: Optional[str] = None):
        """创建 logits margin router"""
        router = LogitsMarginRouter(model_path)
        router_name = name or "logits_margin"
        self.register_router(router_name, router)
        return router_name

    def create_semantic_entropy_router(self, model_path: str, num_samples: int = 5, name: Optional[str] = None):
        """创建 semantic entropy router
        Args:
            model_path: 模型路径
            num_samples: 生成样本数用于计算熵
            name: router名称
        """
        router = SemanticEntropyRouter(model_path, num_samples)
        router_name = name or "semantic_entropy"
        self.register_router(router_name, router)
        return router_name

    def create_max_logits_router(self, name: Optional[str] = None):
        """创建 max logits router"""
        router = MaxLogitsRouter()
        router_name = name or "max_logits"
        self.register_router(router_name, router)
        return router_name

    def create_top10_variance_router(self, name: Optional[str] = None):
        """创建 top-10 variance router"""
        router = Top10VarianceRouter()
        router_name = name or "top10_variance"
        self.register_router(router_name, router)
        return router_name

    def create_coe_router(self, name: Optional[str] = None):
        """创建 CoE router"""
        router = CoERouter()
        router_name = name or "coe"
        self.register_router(router_name, router)
        return router_name

    def create_entropy_router(self, name: Optional[str] = None):
        """创建 entropy router"""
        router = EntropyRouter()
        router_name = name or "entropy"
        self.register_router(router_name, router)
        return router_name

    def create_confidence_margin_router(self, name: Optional[str] = None):
        """创建 confidence margin router"""
        router = ConfidenceMarginRouter()
        router_name = name or "confidence_margin"
        self.register_router(router_name, router)
        return router_name

    def list_routers(self) -> List[str]:
        return list(self.routers.keys())


def create_router_manager() -> RouterManager:
    return RouterManager()


class LogitsMarginRouter(Router):
    """Router based on logits margin (difference between top-2 predictions)"""

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        """Calculate confidence scores based on logits margin

        Higher margin = higher confidence = higher score
        """
        scores = []

        for item in data:
            # Get logits from the data
            logits = item.get("logits", None)

            if logits is None:
                # Fallback: use default score if no logits available
                scores.append(0.5)
                continue

            # Convert to tensor if needed
            if isinstance(logits, np.ndarray):
                logits = torch.tensor(logits, dtype=torch.float32)
            elif not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            # Calculate margin between top-2 logits
            if logits.dim() > 1:
                # If batch dimension exists, take the first one
                logits = logits[0]

            top2_values = torch.topk(logits, k=2).values
            margin = (top2_values[0] - top2_values[1]).item()

            # Normalize margin to [0, 1] using sigmoid
            # Higher margin -> higher confidence
            score = torch.sigmoid(torch.tensor(margin)).item()
            scores.append(score)

        return np.array(scores)


class SemanticEntropyRouter(Router):
    """Router based on semantic entropy of model predictions"""

    def __init__(self, model_path: str, num_samples: int = 5, device: Optional[str] = None):
        self.model_path = model_path
        self.num_samples = num_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def calculate_semantic_entropy(self, responses: List[str]) -> float:
        """Calculate semantic entropy from multiple responses

        Lower entropy = more consistent responses = higher confidence
        """
        if len(responses) <= 1:
            return 0.0

        # Simple semantic clustering: group similar responses
        # Use normalized edit distance as similarity metric
        from difflib import SequenceMatcher

        clusters = []
        for response in responses:
            # Find most similar cluster
            best_cluster_idx = -1
            best_similarity = 0.0

            for idx, cluster in enumerate(clusters):
                # Calculate similarity with cluster representative
                similarity = SequenceMatcher(None, response.lower(), cluster["representative"].lower()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_idx = idx

            # Add to cluster if similarity > threshold, else create new cluster
            if best_similarity > 0.7 and best_cluster_idx >= 0:
                clusters[best_cluster_idx]["count"] += 1
            else:
                clusters.append({"representative": response, "count": 1})

        # Calculate entropy from cluster distribution
        counts = np.array([c["count"] for c in clusters])
        probs = counts / counts.sum()
        entropy = scipy_entropy(probs)

        return entropy

    def get_router_scores(self, data: List[Dict], model_type: str = "weak", **kwargs) -> np.ndarray:
        """Calculate confidence scores based on semantic entropy

        Lower entropy = higher confidence = higher score
        """
        # Check if all items already have multiple responses
        all_have_responses = all("responses" in item and len(item["responses"]) >= 2 for item in data)

        if not all_have_responses:
            # Batch generate multiple responses for all items
            all_prompts = []
            for item in data:
                instruction = item.get("instruction", "")
                # Repeat each instruction num_samples times
                all_prompts.extend([instruction] * self.num_samples)

            # Single parallel inference call for all prompts
            all_responses = parallel_inference(
                all_prompts,
                max_tokens=512,
                temperature=0.7,  # Use sampling for diversity
                model_name_or_path=self.model_path,
                type=model_type
            )

            # Reshape responses back to (num_items, num_samples)
            responses_per_item = []
            for i in range(len(data)):
                start_idx = i * self.num_samples
                end_idx = start_idx + self.num_samples
                responses_per_item.append(all_responses[start_idx:end_idx])
        else:
            # Use existing responses
            responses_per_item = [item["responses"] for item in data]

        # Calculate semantic entropy for each item
        scores = []
        for responses in responses_per_item:
            entropy = self.calculate_semantic_entropy(responses)
            # Convert entropy to confidence score
            # Lower entropy -> higher confidence
            # Use exponential decay to map entropy to [0, 1]
            score = np.exp(-entropy)
            scores.append(score)

        return np.array(scores)


class MaxLogitsRouter(Router):
    """Router based on maximum logits value"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        """Calculate confidence scores based on maximum logits

        Higher max logits = higher confidence = higher score
        """
        scores = []

        for item in data:
            # Handle tuple format: (logits_array, label)
            if isinstance(item, tuple):
                logits = item[0]  # First element is logits
            else:
                # Handle dict format: {"logits": array}
                logits = item.get("logits", None)
                if logits is None:
                    raise ValueError("MaxLogitsRouter requires 'logits' field in data")

            # Convert to tensor if needed
            if isinstance(logits, np.ndarray):
                logits = torch.tensor(logits, dtype=torch.float32)
            elif not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            # Get maximum logit value
            if logits.dim() > 1:
                logits = logits[0]

            max_logit = torch.max(logits).item()

            # Normalize using sigmoid
            score = torch.sigmoid(torch.tensor(max_logit)).item()
            scores.append(score)

        return np.array(scores)


class Top10VarianceRouter(Router):
    """Router based on variance of top-10 logits"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        """Calculate confidence scores based on top-10 logits variance

        Lower variance = higher confidence = higher score
        """
        scores = []

        for item in data:
            # Handle tuple format: (logits_array, label)
            if isinstance(item, tuple):
                logits = item[0]  # First element is logits
            else:
                # Handle dict format: {"logits": array}
                logits = item.get("logits", None)
                if logits is None:
                    raise ValueError("Top10VarianceRouter requires 'logits' field in data")

            # Convert to tensor if needed
            if isinstance(logits, np.ndarray):
                logits = torch.tensor(logits, dtype=torch.float32)
            elif not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            # Get top-10 logits
            if logits.dim() > 1:
                logits = logits[0]

            k = min(10, len(logits))
            top_k_values = torch.topk(logits, k=k).values

            # Calculate variance
            variance = torch.var(top_k_values).item()

            # Convert variance to confidence score
            # Lower variance -> higher confidence
            # Use negative exponential to map variance to [0, 1]
            score = np.exp(-variance)
            scores.append(score)

        return np.array(scores)


class CoERouter(Router):
    """Router based on Confidence of Expert (CoE) signal"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        """Calculate confidence scores based on CoE signal

        Uses the same logic as the existing coe probe implementation
        """
        scores = []

        for item in data:
            # Handle tuple format: (hidden_states_array, label)
            if isinstance(item, tuple):
                hidden_states = item[0]  # First element is hidden_states
            else:
                # Handle dict format: {"hidden_states": array}
                hidden_states = item.get("hidden_states", None)

            if hidden_states is None:
                scores.append(0.5)
                continue

            # Convert to tensor if needed
            if isinstance(hidden_states, np.ndarray):
                hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
            elif not isinstance(hidden_states, torch.Tensor):
                hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

            # Calculate CoE features (magnitude and angle features)
            mag_features = []
            angle_features = []

            for j in range(hidden_states.shape[0] - 1):
                h_curr = hidden_states[j]
                h_next = hidden_states[j+1]

                # Magnitude feature
                mag = torch.norm(h_curr, p=2) + torch.norm(h_next, p=2)
                mag_features.append(mag)

                # Angle feature (cosine similarity)
                cos_sim = F.cosine_similarity(h_curr, h_next, dim=0)
                angle_features.append(cos_sim)

            # Combine features
            coe_features = torch.cat([
                torch.stack(mag_features),
                torch.stack(angle_features)
            ], dim=0)

            # Use mean of combined features as confidence score
            score = torch.sigmoid(torch.mean(coe_features)).item()
            scores.append(score)

        return np.array(scores)


class EntropyRouter(Router):
    """Router based on prediction entropy"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        """Calculate confidence scores based on prediction entropy

        Lower entropy = higher confidence = higher score
        """
        scores = []

        for item in data:
            # Handle tuple format: (logits_array, label)
            if isinstance(item, tuple):
                logits = item[0]  # First element is logits
            else:
                # Handle dict format: {"logits": array}
                logits = item.get("logits", None)
                if logits is None:
                    raise ValueError("EntropyRouter requires 'logits' field in data")

            # Convert to tensor if needed
            if isinstance(logits, np.ndarray):
                logits = torch.tensor(logits, dtype=torch.float32)
            elif not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            # Calculate probabilities
            if logits.dim() > 1:
                logits = logits[0]

            probs = torch.softmax(logits, dim=0)

            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            # Convert entropy to confidence score
            # Lower entropy -> higher confidence
            # Normalize entropy by log(vocab_size) for better scaling
            max_entropy = np.log(len(logits))
            normalized_entropy = entropy / max_entropy
            score = 1.0 - normalized_entropy
            scores.append(max(0.0, min(1.0, score)))

        return np.array(scores)


class ConfidenceMarginRouter(Router):
    """Router based on confidence margin (max_prob - second_max_prob)"""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        """Calculate confidence scores based on probability margin

        Higher margin = higher confidence = higher score
        """
        scores = []

        for item in data:
            # Handle tuple format: (logits_array, label)
            if isinstance(item, tuple):
                logits = item[0]  # First element is logits
            else:
                # Handle dict format: {"logits": array}
                logits = item.get("logits", None)
                if logits is None:
                    raise ValueError("ConfidenceMarginRouter requires 'logits' field in data")

            # Convert to tensor if needed
            if isinstance(logits, np.ndarray):
                logits = torch.tensor(logits, dtype=torch.float32)
            elif not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32)

            # Calculate probabilities
            if logits.dim() > 1:
                logits = logits[0]

            probs = torch.softmax(logits, dim=0)

            # Get top-2 probabilities
            top2_probs = torch.topk(probs, k=2).values
            margin = (top2_probs[0] - top2_probs[1]).item()

            # Margin is already in [0, 1] range, so we can use it directly
            scores.append(margin)

        return np.array(scores)


def get_available_probe_types() -> List[str]:
    return list(ProbeRouter.PROBE_TYPES.keys())

