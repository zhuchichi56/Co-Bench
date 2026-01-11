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



class EmbeddingMLPNet(nn.Module):
    """Simple MLP for routing based on pre-computed query embeddings."""

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or []
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EmbeddingMLPRouter(Router):
    """
    Routing with query-level embeddings (same idea as LLMRouter MLP).

    Input: each sample contains a `query_embedding` or `embedding` field
    (list / np.ndarray / torch.Tensor).
    Output: sigmoid score (larger means the model is more likely correct / more confident),
    aligned with the probe router score direction.
    """

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model, self.metadata = self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        metadata = ckpt.get("metadata", {})

        # metadata keys we care about
        input_dim = metadata.get("input_dim")
        hidden_dims = metadata.get("hidden_dims") or metadata.get("mlp_hidden_dims")
        dropout = metadata.get("dropout", 0.1)

        # state dict field name
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

        if input_dim is None:
            # Try to infer from weights (first Linear in_features)
            for k, v in state_dict.items():
                if "net.0.weight" in k or "layers.0.weight" in k:
                    input_dim = v.shape[1]
                    break
            if input_dim is None:
                raise ValueError("input_dim not found in metadata or state_dict; please include it in checkpoint metadata.")

        model = EmbeddingMLPNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        model.load_state_dict(state_dict, strict=False)
        return model, metadata

    @staticmethod
    def _to_tensor(embed: Any) -> torch.Tensor:
        if isinstance(embed, torch.Tensor):
            return embed.float()
        if isinstance(embed, np.ndarray):
            return torch.tensor(embed, dtype=torch.float32)
        return torch.tensor(embed, dtype=torch.float32)

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        if len(data) == 0:
            return np.array([])

        embeds = []
        for item in data:
            emb = item.get("query_embedding")
            if emb is None:
                emb = item.get("embedding")
            if emb is None:
                # fallback: zero vector with input_dim
                emb = torch.zeros(self.model.net[0].in_features if isinstance(self.model.net[0], nn.Linear) else self.metadata.get("input_dim", 1))
            embeds.append(self._to_tensor(emb))

        X = torch.stack(embeds).to(self.device)
        with torch.no_grad():
            logits = self.model(X).squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
        return scores


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

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(self.model, "resize_token_embeddings"):
                self.model.resize_token_embeddings(len(self.tokenizer))

        self.sep_token = "<SEP>"

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        self.model.eval()
        scores = []

        with torch.no_grad():
            for item in data:
                question = item.get("instruction", "")
                # Format as: Question only
                text = f"{question}"

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
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None or len(hidden_dims) == 0:
            # Single linear layer (original behavior)
            self.net = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer MLP
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)

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
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.1, **kwargs):
        super().__init__()
        if hidden_dims is None or len(hidden_dims) == 0:
            # Single linear layer (original behavior)
            self.fc = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer MLP
            layers = []
            prev_dim = input_dim
            print(f"hidden_dims:{hidden_dims}")
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x.mean(dim=1))


class MaxProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.1, **kwargs):
        super().__init__()
        if hidden_dims is None or len(hidden_dims) == 0:
            # Single linear layer (original behavior)
            self.fc = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer MLP
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x.max(dim=1).values)


class DynamicFusionProbe(nn.Module):
    """Probe that dynamically fuses signals from each layer, supports softmax and Dirichlet methods"""
    def __init__(self, input_dim: int, num_layers: int, output_dim: int = 1, probe_type: str = "softmax"):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.probe_type = probe_type

        if probe_type == "softmax":
            # Original method: learnable weight parameters for each layer
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        elif probe_type == "dirichlet":
            # Fixed global concentration parameters
            self.concentration_logits = nn.Parameter(torch.ones(num_layers))  # Learn log(α)
            self.global_concentration = nn.Parameter(torch.tensor(1.0))  # Learn β₀
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        # Final classifier
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states, return_uncertainty=False, return_weights=False):
        """
        Args:
            hidden_states: [batch_size, num_layers, hidden_dim]
            return_uncertainty: Whether to return uncertainty metric (only effective for Dirichlet)
            return_weights: Whether to return alpha weights [batch_size, num_layers]
        Returns:
            logits: [batch_size, output_dim]
            uncertainty: (optional) Uncertainty metric
            weights: (optional) Alpha weights [batch_size, num_layers]
        """
        batch_size = hidden_states.size(0)

        if self.probe_type == "softmax":
            # Original method: simple softmax weights
            weights = torch.softmax(self.layer_weights, dim=0)  # [num_layers]
            weights_expanded = weights.unsqueeze(0).unsqueeze(-1)  # [1, num_layers, 1]
            fused_features = torch.sum(hidden_states * weights_expanded, dim=1)  # [batch_size, hidden_dim]

            logits = self.classifier(fused_features)

            # Build return value
            result = [logits]
            if return_uncertainty:
                result.append(None)  # Original method doesn't provide uncertainty
            if return_weights:
                # Return weights for each sample [batch_size, num_layers]
                result.append(weights.unsqueeze(0).expand(batch_size, -1))

            if len(result) == 1:
                return result[0]
            return tuple(result)

        elif self.probe_type == "dirichlet":
            # Dirichlet method: sample weights from Dirichlet distribution
            # Fixed global concentration parameters
            base_concentration = torch.softmax(self.concentration_logits, dim=0)  # [num_layers]
            concentration = torch.exp(self.global_concentration) * base_concentration  # [num_layers]
            # Expand to batch dimension for unified processing
            concentration = concentration.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_layers]

            if self.training:
                # During training: sample from Dirichlet distribution
                # Use corresponding concentration for each sample
                weights_list = []
                uncertainty_list = []
                for i in range(batch_size):
                    dirichlet_dist = Dirichlet(concentration[i])
                    sample_weights = dirichlet_dist.rsample()  # [num_layers]
                    weights_list.append(sample_weights)
                    uncertainty_list.append(dirichlet_dist.entropy())  # scalar

                weights = torch.stack(weights_list, dim=0)  # [batch_size, num_layers]
                uncertainty = torch.stack(uncertainty_list, dim=0)  # [batch_size]
                weights_for_fusion = weights.unsqueeze(-1)  # [batch_size, num_layers, 1]
            else:
                # During inference: use expected value ᾱ_l = β_l / Σβ_j
                beta_0 = concentration.sum(dim=1, keepdim=True)  # [batch_size, 1] - β₀
                weights = concentration / beta_0  # [batch_size, num_layers] - ᾱ
                weights_for_fusion = weights.unsqueeze(-1)  # [batch_size, num_layers, 1]

                # Calculate uncertainty: use β₀ = Σβ_j
                # Large β₀ indicates high confidence (low uncertainty)
                # Small β₀ indicates low confidence (high uncertainty)
                # Use negative log as uncertainty metric
                uncertainty = -torch.log(beta_0.squeeze(1))  # [batch_size]

            # Weighted fusion
            fused_features = torch.sum(hidden_states * weights_for_fusion, dim=1)  # [batch_size, hidden_dim]
            logits = self.classifier(fused_features)

            # Build return value
            result = [logits]
            if return_uncertainty:
                result.append(uncertainty)
            if return_weights:
                result.append(weights)  # [batch_size, num_layers]

            if len(result) == 1:
                return result[0]
            return tuple(result)



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
           
            num_layers = metadata.get("num_layers", 32)
            probe_method = "softmax" if self.probe_type == "dynamic_softmax" else "dirichlet"
            model = model_class(input_dim, num_layers, output_dim, probe_method)
            
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
        elif self.probe_type == "pca_conv":
            model = model_class(input_dim, output_dim)
            model.load_state_dict(model_state)
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
        """
        Extract features from input data based on probe type
        
        Note: For dynamic probes (dynamic_softmax, dynamic_dirichlet), 
        this method is not used; get_router_scores handles them directly.
        """
        features = []
        for i, item in enumerate(data):
            # Handle tuple format data
            if isinstance(item, tuple):
                # Assume first element is hidden_states, second is label
                hidden_states = item[0]  # numpy.ndarray
            else:
                # Handle dict format data
                hidden_states = item.get("hidden_states", [])
            
            # Convert numpy array to torch tensor
            if isinstance(hidden_states, np.ndarray):
                hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
            elif not isinstance(hidden_states, torch.Tensor):
                hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

            if self.probe_type == "hs_last_mlp" or self.probe_type == "hs_mlp":
                feat = hidden_states[-1]
            elif self.probe_type in ["coe_dual_mlp", "coe_c_scalar", "coe_r_scalar"]:
                # Compute CoE (Coefficient of Evolution) features
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
                feat = hidden_states

            features.append(feat.unsqueeze(0))

        return torch.cat(features, dim=0)

    def get_router_scores(self, data: List[Dict], **kwargs) -> np.ndarray:
        """
        Calculate router scores

        Args:
            data: Input data list
            **kwargs: Other parameters

        Returns:
            np.ndarray of scores
        """
        self.model.eval()
        
        # For dynamic probes, process hidden_states directly (no extract_features needed)
        if self.probe_type in ["dynamic_softmax", "dynamic_dirichlet"]:
            features = []
            for item in data:
                # Handle tuple format data
                if isinstance(item, tuple):
                    hidden_states = item[0]  # numpy.ndarray
                else:
                    # Handle dict format data
                    hidden_states = item.get("hidden_states", [])

                # Convert numpy array to torch tensor
                if isinstance(hidden_states, np.ndarray):
                    hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
                elif not isinstance(hidden_states, torch.Tensor):
                    hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

                features.append(hidden_states.unsqueeze(0))

            features = torch.cat(features, dim=0).to(self.device)
            
            with torch.no_grad():
                # Use deterministic inference
                logits = self.model(features)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                scores = torch.sigmoid(logits).cpu().numpy()

            return scores
        
        # For other probe types, use extract_features
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

    def create_logits_margin_router(self, model_path: str, name: Optional[str] = None):
        """Create a logits-margin router."""
        router = LogitsMarginRouter(model_path)
        router_name = name or "logits_margin"
        self.register_router(router_name, router)
        return router_name

    def create_semantic_entropy_router(self, model_path: str, num_samples: int = 5, name: Optional[str] = None):
        """Create a semantic-entropy router.
        Args:
            model_path: model path
            num_samples: number of samples to generate for entropy estimation
            name: router name
        """
        router = SemanticEntropyRouter(model_path, num_samples)
        router_name = name or "semantic_entropy"
        self.register_router(router_name, router)
        return router_name

    def create_max_logits_router(self, name: Optional[str] = None):
        """Create a max-logits router."""
        router = MaxLogitsRouter()
        router_name = name or "max_logits"
        self.register_router(router_name, router)
        return router_name

    def create_top10_variance_router(self, name: Optional[str] = None):
        """Create a top-10 variance router."""
        router = Top10VarianceRouter()
        router_name = name or "top10_variance"
        self.register_router(router_name, router)
        return router_name

    def create_coe_router(self, name: Optional[str] = None):
        """Create a CoE router."""
        router = CoERouter()
        router_name = name or "coe"
        self.register_router(router_name, router)
        return router_name

    def create_entropy_router(self, name: Optional[str] = None):
        """Create an entropy router."""
        router = EntropyRouter()
        router_name = name or "entropy"
        self.register_router(router_name, router)
        return router_name

    def create_embedding_mlp_router(self, checkpoint_path: str, name: Optional[str] = None):
        """Create an MLP router based on query embeddings."""
        router = EmbeddingMLPRouter(checkpoint_path)
        router_name = name or "embedding_mlp"
        self.register_router(router_name, router)
        return router_name

    def create_confidence_margin_router(self, name: Optional[str] = None):
        """Create a confidence-margin router."""
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
