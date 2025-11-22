# Router 使用指南

CoBench 现在支持以下 Router 类型：

## 1. Probe Router
基于隐藏状态的探针路由器

```yaml
router:
  router_type: "probe"
  checkpoint_path: "probe_save/your_probe.pt"
  probe_type: "mean"  # 可选: hs_mlp, mean, max, transformer, etc.
```

## 2. Self-Questioning Router
基于模型自我评估的路由器

```yaml
router:
  router_type: "self_questioning"
  model_path: null  # 默认使用 weak_model_path
```

## 3. DeBERTa Router
基于 DeBERTa 的路由器

```yaml
router:
  router_type: "deberta"
  model_path: "microsoft/deberta-v3-base"
```

## 4. Trained DeBERTa Router
使用训练过的 DeBERTa 模型

```yaml
router:
  router_type: "trained_deberta"
  model_path: "path/to/trained/deberta"
```

## 5. LLM Router
基于 LLM 的难度评估路由器

```yaml
router:
  router_type: "llm"
  model_path: null  # 默认使用 weak_model_path
```

## 6. Logits Margin Router (NEW!)
基于 logits 间隔的路由器

**原理**: 计算模型输出 logits 的前两个最高值之间的间隔。间隔越大说明模型越自信，分数越高。

```yaml
router:
  router_type: "logits_margin"
  model_path: null  # 仅用于标识，实际使用数据中的 logits
```

**使用说明**:
- 需要在模型推理时保存 logits 信息
- 数据格式需要包含 `logits` 字段
- 分数计算: `score = sigmoid(top1_logit - top2_logit)`

## 7. Semantic Entropy Router (NEW!)
基于语义熵的路由器

**原理**: 通过多次采样生成多个回答，计算语义一致性的熵。熵越低说明回答越一致，模型越自信。

```yaml
router:
  router_type: "semantic_entropy"
  model_path: null  # 默认使用 weak_model_path
  num_samples: 5    # 生成样本数量
```

**使用说明**:
- 会对每个问题生成 `num_samples` 个回答
- 使用编辑距离对回答进行聚类
- 计算聚类分布的熵作为不确定性度量
- 分数计算: `score = exp(-entropy)`
- **注意**: 此 router 会显著增加推理时间

## 配置示例

### Logits Margin 配置示例
```yaml
# config_logits_margin.yaml
data_dir: "data"
output_dir: "results"
metric_results_dir: "metric_results/logits_margin"

inference:
  weak_model_path: "/path/to/your/model"
  max_tokens: 2048
  temperature: 0.0

router:
  router_type: "logits_margin"
  model_path: null
```

### Semantic Entropy 配置示例
```yaml
# config_semantic_entropy.yaml
data_dir: "data"
output_dir: "results"
metric_results_dir: "metric_results/semantic_entropy"

inference:
  weak_model_path: "/path/to/your/model"
  max_tokens: 512
  temperature: 0.7  # 用于生成多样化的样本

router:
  router_type: "semantic_entropy"
  model_path: null
  num_samples: 5  # 每个问题生成5个回答
```

## 使用方法

### 1. 通过 Pipeline 运行

```bash
# 使用 Logits Margin Router
python src/pipeline.py evaluate_complete_pipeline \
  --hidden_states_file your_hidden_states.pt \
  --datasets "['aime24']" \
  --config config_logits_margin.yaml

# 使用 Semantic Entropy Router
python src/pipeline.py evaluate_complete_pipeline \
  --hidden_states_file your_hidden_states.pt \
  --datasets "['aime24']" \
  --config config_semantic_entropy.yaml
```

### 2. Python 代码调用

```python
from pipeline import RouterEvaluationPipeline
from config import PipelineConfig

# 加载配置
config = PipelineConfig.from_yaml()

# 修改为 logits margin router
config.router.router_type = "logits_margin"

# 或修改为 semantic entropy router
config.router.router_type = "semantic_entropy"
config.router.num_samples = 10  # 增加采样数

# 运行评估
pipeline = RouterEvaluationPipeline(config)
results = pipeline.evaluate_complete_pipeline(
    hidden_states_file="your_data.pt",
    datasets=["aime24"]
)
```

## 性能对比

| Router Type | 推理速度 | 准确性 | 所需数据 |
|------------|---------|--------|---------|
| Probe | ⚡⚡⚡ 极快 | ⭐⭐⭐ 高 | Hidden states |
| Self-Questioning | ⚡⚡ 中等 | ⭐⭐ 中等 | 问题文本 |
| DeBERTa | ⚡⚡ 中等 | ⭐⭐⭐ 高 | 问题+回答 |
| LLM | ⚡ 较慢 | ⭐⭐ 中等 | 问题文本 |
| **Logits Margin** | ⚡⚡⚡ 极快 | ⭐⭐⭐ 高 | **Logits** |
| **Semantic Entropy** | ⚡ 很慢 | ⭐⭐⭐⭐ 很高 | 问题文本 |

## 注意事项

### Logits Margin Router
- ✅ 无需额外推理，速度快
- ✅ 基于模型内部置信度，准确性高
- ⚠️ 需要在推理时保存 logits
- ⚠️ 对分类任务效果最好

### Semantic Entropy Router
- ✅ 捕获语义层面的不确定性
- ✅ 对开放式生成任务效果好
- ⚠️ 需要多次推理，速度慢（时间 ≈ num_samples × 单次推理）
- ⚠️ 依赖于语义相似度计算的准确性

## 数据格式要求

### Logits Margin 所需数据格式
```python
{
    "instruction": "问题文本",
    "generated_response": "模型回答",
    "logits": [0.1, 0.2, 0.5, ...]  # 必需字段
}
```

### Semantic Entropy 所需数据格式
```python
{
    "instruction": "问题文本",
    # 可选: 如果已有多个回答
    "responses": ["回答1", "回答2", ...]
}
```

## 调试建议

1. **Logits Margin 分数过低**: 检查 logits 是否正确归一化
2. **Semantic Entropy 分数都接近**: 增加 `num_samples` 或调整 temperature
3. **推理速度慢**: Semantic Entropy 考虑减少 `num_samples`