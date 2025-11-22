from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import os
import yaml


@dataclass
class InferenceConfig:
    """Inference-related configuration parameters"""

    # Model paths
    strong_model_path: str = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/models/Qwen2.5-14B-Instruct"
    weak_model_path: str = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/models/Qwen2.5-7B-Instruct"

    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True

    # Server configuration
    base_port: int = 8000
    strong_gpu_ids: List[int] = field(default_factory=lambda: [0, 1])
    weak_gpu_ids: List[int] = field(default_factory=lambda: [2, 3])

    # GPT API configuration
    openai_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("sk-UfaLUh9hblwmlr0p843eF3A3Ab324a878724484fA8B45c89"))
    openai_api_base: Optional[str] = field(default_factory=lambda: os.environ.get("https://api.ai-gaochao.cn/v1'"))

    # Parallel inference settings
    max_workers: int = 32
    batch_size: int = 8

    # Template settings
    template_type: str = "default"
    system_prompt: str = "You are a helpful AI assistant."

    def get_server_urls(self, model_type: str = "weak") -> List[str]:
        """Generate server URLs based on model type"""
        if model_type == "strong":
            gpu_ids = self.strong_gpu_ids
        else:  # weak
            gpu_ids = self.weak_gpu_ids
        return [f"http://localhost:{self.base_port + gpu_id}" for gpu_id in gpu_ids]

    def get_model_path(self, model_type: str = "weak") -> str:
        """Get model path based on type"""
        if model_type == "strong":
            return self.strong_model_path
        else:  # weak
            return self.weak_model_path


@dataclass
class RouterConfig:
    """Router-specific configuration"""

    router_type: str = "probe" # "probe", "self_questioning", "deberta", "trained_deberta", "llm", "logits_margin", "semantic_entropy", "max_logits", "top10_variance", "coe", "entropy", "confidence_margin"

    # Probe router settings
    checkpoint_path: Optional[str] = "probe_save/mixed_mmlu_full_numina_cot_5k_balanced_probe_mean.pt"
    probe_type: str = "mean"
    # hidden_states_file: Optional[str] = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/logits/Qwen2.5-7B-Instruct_aime24.pt"

    # Model router settings (self_questioning, deberta, trained_deberta, llm, logits_margin, semantic_entropy)
    model_path: Optional[str] = None

    # Semantic entropy specific settings
    num_samples: int = 5  # Number of samples for semantic entropy calculation

    def to_dict(self, inference_config=None) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility"""
        config_dict = {"type": self.router_type}

        if self.router_type == "probe":
            config_dict.update({
                "checkpoint_path": self.checkpoint_path,
                "probe_type": self.probe_type
            })
        elif self.router_type in ["self_questioning", "deberta", "trained_deberta", "llm", "logits_margin", "semantic_entropy"]:
            # Use weak model if not specified
            model_path = self.model_path
            if model_path is None and inference_config is not None:
                model_path = inference_config.weak_model_path
            config_dict["model_path"] = model_path

            # Add semantic entropy specific config
            if self.router_type == "semantic_entropy":
                config_dict["num_samples"] = self.num_samples
        elif self.router_type in ["max_logits", "top10_variance", "coe", "entropy", "confidence_margin"]:
            # Logits-based routers don't need additional config
            pass

        return config_dict


@dataclass
class TrainingConfig:
    """Training-related configuration"""

    # Probe training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    # Reward model training
    reward_model_name: str = "microsoft/deberta-v3-base"
    reward_output_dir: str = "reward_model"
    logits_output_dir: str = "/HOME/sustc_ghchen/sustc_ghchen_4/HDD_POOL/logits/"
    probe_save_path: str = "probe_save"
    seed: int = 42
    
    


@dataclass
class PipelineConfig:
    """Pipeline execution configuration"""

    # Data directories
    data_dir :str ="data"
    output_dir: str = "results/"
    metric_results_dir: str = "metric_results"

    # Evaluation parameters
    recovery_rate_band: Tuple[float, float] = (0.91, 0.92)
    call_rate_param: float = 0.1

    # Default datasets
    # default_datasets: List[str] = field(default_factory=lambda: ["aime24"])

    # Sub-configurations
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls):
        """自动检测机器并加载对应配置"""
        def detect_machine():
            """检测当前机器类型"""
            # 方法1: 检查hostname
            import socket
            hostname = socket.gethostname()

            # 方法2: 检查特定路径是否存在
            if os.path.exists("/data1/wwx/models/models"):
                return "A"
            elif os.path.exists("/volume/pt-train/users/wzhang/ghchen/zh/CoBench"):  # 替换为机器B的特征路径
                return "B"

            # 方法3: 检查环境变量（备用）
            if "MACHINE_ID" in os.environ:
                return os.environ["MACHINE_ID"]

            # 默认为A
            return "A"

        machine_id = detect_machine()
        config_file = f"../config_{machine_id}.yaml"

        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在，使用默认配置")
            return cls()

        print(f"自动检测为机器 {machine_id}，加载配置: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        inference_config = InferenceConfig(**data.get('inference', {}))
        router_config = RouterConfig(**data.get('router', {}))
        training_config = TrainingConfig(**data.get('training', {}))

        pipeline_data = {k: v for k, v in data.items()
                        if k not in ['inference', 'router', 'training']}

        return cls(
            inference=inference_config,
            router=router_config,
            training=training_config,
            **pipeline_data
        )


