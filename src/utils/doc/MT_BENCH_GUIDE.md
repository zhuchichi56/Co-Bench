# MT-Bench 评测指南

MT-Bench评测已集成到主pipeline中，支持完整的多轮对话评估流程。

## 快速开始

### 方法1: 使用统一的 get_task_score 接口 (推荐)

```bash
# Python 调用
python -c "
from pipeline import get_task_score
from config import PipelineConfig

config = PipelineConfig.from_yaml()
result = get_task_score(config, 'mt-bench', judge_model='gpt-4o')
print(result)
"
```

### 方法2: 使用命令行

```bash
python pipeline.py get_task_score --task mt-bench --judge_model gpt-4o
```

### 方法3: 旧版接口 (已弃用)

```bash
python pipeline.py evaluate_mt_bench_pipeline config.yaml
```

## 多轮对话处理

### 输入格式
```json
{
  "question_id": 81,
  "category": "writing",
  "turns": [
    "Compose an engaging travel blog post about a recent trip to Hawaii...",
    "Rewrite your previous response. Start every sentence with the letter A."
  ]
}
```

### 推理流程（完全复现原版MT-Bench）

**第一轮对话**:
```
使用模型专用conversation template格式化:
<s>[INST] Compose an engaging travel blog post... [/INST]

模型输出: "Hawaii is a tropical paradise..."
```

**第二轮对话**:
```
使用完整对话历史格式化:
<s>[INST] Compose an engaging travel blog post... [/INST]Hawaii is a tropical paradise...</s>
<s>[INST] Rewrite your previous response. Start every sentence with the letter A. [/INST]

模型输出: "Absolutely stunning, Hawaii is a tropical paradise..."
```

### 关键实现细节

1. **Conversation Template**: 自动检测模型类型并使用对应的模板格式
   - Llama: `<s>[INST] ... [/INST]`
   - Qwen: `<|im_start|>user\n...<|im_end|>\n`
   - GPT: 直接使用messages格式

2. **特殊Token处理**: 自动清理输出中的特殊tokens
   - 移除 `<|im_end|>`, `</s>`, `<s>`, `[/INST]` 等
   - 移除 `Assistant:` 前缀

3. **温度控制**: 与原版完全一致的分类温度配置

### 输出格式
```json
{
  "question_id": 81,
  "choices": [
    {
      "turns": [
        "Hawaii is a tropical paradise...",           // 第一轮回答
        "Absolutely stunning, Hawaii is a tropical..." // 第二轮回答
      ]
    }
  ]
}
```

## 详细用法

### 方式1: 使用统一接口 (推荐)

```bash
# 基本用法
python pipeline.py get_task_score mt-bench

# 指定judge模型
python pipeline.py get_task_score mt-bench --judge_model gpt-4o-mini

# 完整参数
python pipeline.py get_task_score mt-bench \
  --judge_model gpt-4o \
  --question_file data/llmjudge/mt_bench/question.jsonl \
  --ref_answer_file data/llmjudge/mt_bench/reference.jsonl
```

### 方式2: 旧版命令 (已弃用)

```bash
# 基本评测
python pipeline.py evaluate_mt_bench_pipeline config.yaml

# 指定问题文件
python pipeline.py evaluate_mt_bench_pipeline config.yaml \
  --question_file data/llmjudge/mt_bench/question.jsonl

# 使用不同的评判模型
python pipeline.py evaluate_mt_bench_pipeline config.yaml \
  --judge_model gpt-4o \
  --question_file data/llmjudge/mt_bench/question.jsonl

# 包含参考答案
python pipeline.py evaluate_mt_bench_pipeline config.yaml \
  --ref_answer_file data/llmjudge/mt_bench/reference_answer.jsonl
```

## 代码调用

```python
from pipeline import RouterEvaluationPipeline
from config import PipelineConfig

# 加载配置
config = PipelineConfig.from_yaml("config.yaml")
pipeline = RouterEvaluationPipeline(config)

# 运行MT-Bench评测
results = pipeline.evaluate_mt_bench(
    question_file="data/llmjudge/mt_bench/question.jsonl",
    judge_model="gpt-4o",
    ref_answer_file=None  # 可选
)

# 查看结果
print(f"小模型平均分: {results['small_avg']:.2f}")
print(f"大模型平均分: {results['large_avg']:.2f}")
print(f"有效评分: {results['small_valid']}/{results['questions']}")
```

## 输出结果

### 评测结果结构
```python
{
    "questions": 80,              # 总问题数
    "small_model": "model_path",  # 小模型路径
    "large_model": "model_path",  # 大模型路径
    "judge_model": "gpt-4o",      # 评判模型
    "small_scores": [7.0, 8.5, 6.0, ...],  # 小模型详细分数
    "large_scores": [8.0, 9.0, 7.5, ...],  # 大模型详细分数
    "small_avg": 6.8,             # 小模型平均分
    "large_avg": 8.2,             # 大模型平均分
    "small_valid": 78,            # 小模型有效评分数
    "large_valid": 80             # 大模型有效评分数
}
```

### 评分标准
- **评分范围**: 1-10分
- **失败标记**: -1（生成失败或无法评分）
- **评判标准**:
  - 一般问题: 有用性、相关性、准确性、深度、创造性
  - 数学/编程: 正确性、有用性（需要参考答案）
  - 多轮对话: 重点评估第二轮回答

## 温度配置

不同类别问题使用不同的生成温度：

```python
TEMPERATURE_CONFIG = {
    "writing": 0.7,        # 写作类需要创造性
    "roleplay": 0.7,       # 角色扮演需要灵活性
    "extraction": 0.0,     # 信息提取需要准确性
    "math": 0.0,           # 数学题需要确定性
    "coding": 0.0,         # 编程题需要精确性
    "reasoning": 0.0,      # 推理题需要逻辑性
    "stem": 0.1,           # STEM学科适中随机性
    "humanities": 0.1,     # 人文学科适中随机性
}
```

## 技术细节

### 多轮对话实现
1. **第一轮**: 直接使用问题作为prompt
2. **第二轮**: 构建完整对话历史作为context
3. **格式化**: 使用标准的User/Assistant格式

### 评判流程
1. 根据问题类别选择合适的评判prompt
2. 单轮vs多轮自动检测
3. 数学题自动使用参考答案对比
4. GPT-4o生成评判和评分
5. 正则表达式提取数字分数

### 与现有Pipeline集成
- 复用现有的模型配置和推理基础设施
- 使用统一的配置系统
- 自动处理不同模型的调用（本地VLLM vs GPT API）

## 常见问题

### Q: 如何处理生成失败的情况？
A: 系统会标记为-1分，不计入平均分计算，但会在有效评分统计中体现。

### Q: 多轮对话的评判是否考虑第一轮？
A: 评判会看到完整对话，但会重点关注第二轮回答的质量。

### Q: 支持哪些评判模型？
A: 主要支持GPT-4o系列，也可以尝试其他OpenAI兼容模型。

### Q: 如何自定义评判标准？
A: 可以修改`loss_calculator.py`中的`JUDGE_PROMPTS`字典。

## 示例输出

```
Starting MT-Bench evaluation with judge model: gpt-4o
Evaluating 80 MT-Bench questions
Generating small model responses...
Generating large model responses...
Evaluating small model with LLM-as-a-Judge...
Evaluating large model with LLM-as-a-Judge...
Small model average score: 6.75 (78/80 valid)
Large model average score: 8.20 (80/80 valid)
```

这个集成保持了MT-Bench的所有核心特性，同时完美融入了现有的pipeline架构。