# Router Evaluation Framework

A refactored, modular framework for evaluating language model routing systems with multiple router types and comprehensive metrics.

## Architecture

- **`data.py`**: Extensible dataset handling and model evaluation
- **`router.py`**: Router implementations (probe-based, self-questioning, deberta)
- **`metric.py`**: Evaluation metrics (Reliable AUROC, Adaptive LPM/HPM/MPM)
- **`train_router.py`**: Training interfaces for probe and reward models
- **`pipeline.py`**: Main integration pipeline with Fire CLI

## Quick Start

### 1. 启动 VLLM 服务器
首先启动强模型和弱模型的服务器：

```bash
# 启动弱模型服务器 (GPU 4,5 -> 端口 8004, 8005)
conda activate coe
cd inference



# 启动强模型服务器 (GPU 0,1 -> 端口 8000, 8001)
conda activate coe
cd inference
python start.py \
  --model_path "/volume/pt-train/models/Llama-3.1-8B-Instruct" \
  --base_port 8000 \
  --gpu_list "2,3,4,5"

  
```

### 2. 使用配置文件运行 MMLU 评估

```python
# 创建配置并运行评估
from config import PipelineConfig
from pipeline import RouterEvaluationPipeline

# 使用默认配置 (mmlu 数据集, self_questioning 路由器)
config = PipelineConfig()
pipeline = RouterEvaluationPipeline(config)

# 运行完整评估管道
results = pipeline.evaluate_complete_pipeline()
```

### 3. 命令行快速评估 MMLU

```bash
# 使用 self-questioning 路由器评估 MMLU
python -c "
from config import PipelineConfig
from pipeline import evaluate_pipeline

config = PipelineConfig()
results = evaluate_pipeline(
    config=config,
    small_model_path=config.inference.weak_model_path,
    large_model_path=config.inference.strong_model_path
)
print('Evaluation completed!')
print('Results:', results['metric_results'])
"
```

### 4. 查看评估结果

评估完成后，结果保存在以下位置：

```bash
# 查看模型评估结果
ls results/
├── Qwen2.5-7B-Instruct/     # 弱模型结果
│   └── mmlu.jsonl
├── Qwen2.5-14B-Instruct/    # 强模型结果
│   └── mmlu.jsonl
└── metric_results/          # 评估指标
    └── *.json

# 查看 MMLU 准确率
python -c "
import json
with open('results/Qwen2.5-7B-Instruct/mmlu.jsonl', 'r') as f:
    weak_results = [json.loads(line) for line in f]
    weak_acc = sum(r['score'] for r in weak_results) / len(weak_results)
    print(f'弱模型 MMLU 准确率: {weak_acc:.3f}')

with open('results/Qwen2.5-14B-Instruct/mmlu.jsonl', 'r') as f:
    strong_results = [json.loads(line) for line in f]
    strong_acc = sum(r['score'] for r in strong_results) / len(strong_results)
    print(f'强模型 MMLU 准确率: {strong_acc:.3f}')
"
```

### 5. 其他路由器类型评估

```python
# Probe 路由器 (需要先训练)
config = PipelineConfig()
config.router.router_type = "probe"
config.router.checkpoint_path = "probe_model.ckpt"
config.router.hidden_states_file = "hidden_states.pt"
config.router.probe_type = "pca_conv"

# DeBERTa 路由器
config = PipelineConfig()
config.router.router_type = "deberta"
config.router.model_path = "microsoft/deberta-v3-base"
```

## 配置说明

系统使用 dataclass 配置管理，主要配置项：

```python
@dataclass
class InferenceConfig:
    strong_model_path: str = "/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-14B-Instruct"
    weak_model_path: str = "/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-7B-Instruct"
    max_tokens: int = 2048
    temperature: float = 0.0
    strong_gpu_ids: List[int] = [0, 1]  # 强模型使用的GPU
    weak_gpu_ids: List[int] = [4, 5]    # 弱模型使用的GPU

@dataclass
class PipelineConfig:
    default_datasets: List[str] = ["mmlu"]  # 默认评估数据集
    data_dir: str = "data"
    output_dir: str = "results"
```

## 结果分析

### 1. 评估指标含义

- **AUROC**: 路由器区分难易问题的能力
- **Accuracy**: 路由决策的准确性
- **LPM**: 低调用率时的平均准确率
- **HPM**: 高性能指标 (1 - 目标精度带内的平均调用率)
- **MPM**: 中等性能指标

### 2. 典型结果示例

```json
{
  "metric_results": {
    "mmlu": {
      "reliable_auroc": 0.75,
      "adaptive_lpm": 0.82,
      "adaptive_hpm": 0.91,
      "adaptive_mpm": 0.76,
      "avg_accuracy": 0.84
    }
  }
}
```

### 3. 结果文件说明

```
results/
├── Qwen2.5-7B-Instruct/mmlu.jsonl     # 弱模型每个问题的详细结果
├── Qwen2.5-14B-Instruct/mmlu.jsonl    # 强模型每个问题的详细结果
└── metric_results/
    ├── summary_mmlu.json               # 汇总指标
    └── router_scores_mmlu.npy          # 路由器分数
```

## Router Types

### 1. Probe Router
Uses trained probe models on hidden states to predict routing decisions.

**Available probe types:**
- `pca_conv`: PCA + Conv1D probe
- `hs_last_mlp`: Last hidden state MLP
- `coe_dual_mlp`: Combined embedding MLP
- `mean`: Mean pooling probe
- `max`: Max pooling probe
- `mean+max`: Mean+Max combined probe
- `transformer`: Transformer encoder probe

### 2. Self-Questioning Router
Asks the small model to self-assess confidence in its answer.

### 3. Deberta Router
Uses a trained DeBERTa model to classify question-answer pairs.

## Adding Custom Datasets

```python
from data import register_custom_dataset

# Register new dataset
register_custom_dataset(
    name="my_dataset",
    dataset_type="math",  # or "mmlu", "general"
    file_path="data/my_dataset.jsonl"
)
```

Dataset format (JSONL):
```json
{"instruction": "Question text", "response": "Expected answer"}
```

## Metrics

### Reliable Metrics
- **AUROC**: Area under ROC curve for router reliability
- **Accuracy**: Router decision accuracy

### Adaptive Metrics
- **LPM**: Low call rate (≤p%) average accuracy
- **HPM**: High performance metric (1 - avg call rate in target band)
- **MPM**: Medium performance metric (remaining accuracy)

## Pipeline Flow

1. **Model Evaluation**: Evaluate small/large models on datasets → get scores
2. **Router Scoring**: Generate router scores for each test sample
3. **Metric Calculation**: Calculate Reliable & Adaptive metrics
4. **Results Saving**: Save detailed results and plots

## File Structure

```
eval/
├── data.py              # Dataset handling
├── router.py            # Router implementations
├── metric.py            # Evaluation metrics
├── train_router.py      # Training interfaces
├── pipeline.py          # Main pipeline (Fire CLI)
├── data/               # Dataset files
│   ├── mmlu.jsonl
│   ├── math.jsonl
│   └── ...
└── results/            # Output results
    ├── model_name/
    │   └── dataset.jsonl
    └── metric_results/
        └── summary.json
```

## Key Features

- **Modular Design**: Clean separation of concerns
- **Extensible**: Easy to add new datasets and router types
- **No Redundant Saves**: Only saves essential results
- **Fire CLI**: Simple command-line interface
- **Comprehensive**: All probe types from original implementation
- **Efficient**: Minimal memory usage and fast evaluation

## 修改说明
1.  metrics.py: line 46 
- calculate函数下target_accuracy_band 将绝对acc转化为recovery rate(router_acc - small_acc)/(large_acc - small_acc)
- HPM经常为0修复 -- 将sweep决策阈值改为sweep call rate （先升序排序，再取对应百分比call rate调用大模型，若遇到相同分数所占百分比大于call rate，随机选择对应比例）
- pipeline.py line 147 N/A修复

2. train_router.py的complete_probe_training_pipeline 
- 数据导入修复
- 训练probe 只导入小模型hiden state
- 修改 hidden state提取 （coe方法待补充/优化）

3. 修复 router.py MeanMaxProbe 取特征维度不一致

4. train_router.py加速；直接提取sentence的hidden states