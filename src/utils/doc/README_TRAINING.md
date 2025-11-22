# CoBench Training Extensions

This document describes the training capabilities added to the CoBench framework for creating custom routers.

## Overview

Two new training methods have been added to support router training:

1. **DeBERTa Router Training** - Train a DeBERTa model for question classification
2. **LLM Router Training** - Train a language model for difficulty assessment

## File Structure

```
src/
├── cmd/                           # Training scripts
│   ├── run_train_deberta.sh      # DeBERTa 4-GPU training
│   └── run_train_llm.sh          # LLM 4-GPU training
├── data/                         # Training data
│   ├── sample_data.jsonl         # DeBERTa training sample
│   └── sample_alpaca_data.jsonl  # LLM training sample
├── train_deberta.py              # DeBERTa training script
├── train_llm.py                  # LLM training script
└── config_router_examples.yaml  # Configuration examples
```

## Training Scripts

### 1. DeBERTa Router Training

**Purpose**: Train a DeBERTa model to classify questions based on difficulty.

**Input Format**: JSONL with `question`, `llm_id`, and `label` fields
```json
{"question": "What is the capital of France?", "llm_id": "gpt-4", "label": 1}
```

**Usage**:
```bash
# Basic usage
cd cmd && ./run_train_deberta.sh

# With custom parameters
cd cmd && ./run_train_deberta.sh data/my_data.jsonl "" microsoft/deberta-v3-base 2 16 2e-5 3 my_output
```

**Parameters**:
- `data_path`: Training data file (default: data/sample_data.jsonl)
- `val_data_path`: Validation data file (optional)
- `model_name`: Pre-trained model (default: microsoft/deberta-v3-base)
- `num_labels`: Number of classes (default: 2)
- `batch_size`: Training batch size (default: 16)
- `learning_rate`: Learning rate (default: 2e-5)
- `epochs`: Training epochs (default: 3)
- `output_dir`: Output directory (default: deberta_checkpoints)

### 2. LLM Router Training

**Purpose**: Train a language model for question difficulty assessment.

**Input Format**: JSONL with `instruction` and `response` fields (Alpaca format)
```json
{"instruction": "What is the capital of France?", "response": "The capital of France is Paris..."}
```

**Usage**:
```bash
# Basic usage
cd cmd && ./run_train_llm.sh

# With custom parameters
cd cmd && ./run_train_llm.sh data/my_alpaca_data.jsonl /path/to/qwen2.5-0.5b my_output 4 64 2e-5 3 512
```

**Parameters**:
- `data_path`: Training data file (default: data/sample_alpaca_data.jsonl)
- `model_path`: Base model path (default: Qwen2.5-0.5B)
- `output_dir`: Output directory (auto-generated if not provided)
- `batch_size`: Per-device batch size (default: 4)
- `global_batch_size`: Global batch size (default: 64)
- `learning_rate`: Learning rate (default: 2e-5)
- `epochs`: Training epochs (default: 3)
- `max_length`: Maximum sequence length (default: 512)

## Integration with CoBench

### Router Configuration

Update your configuration to use the trained routers:

```yaml
# For trained DeBERTa router
router:
  router_type: "trained_deberta"
  model_path: "deberta_checkpoints/checkpoint_epoch_3"

# For trained LLM router
router:
  router_type: "llm"
  model_path: "output/sft/Qwen2.5-0.5B"
```

### Supported Router Types

- `probe`: Original probe-based router
- `self_questioning`: Self-assessment router
- `deberta`: Original DeBERTa router
- `trained_deberta`: **NEW** - Trained DeBERTa router
- `llm`: **NEW** - Trained LLM router

## Requirements

### DeBERTa Training
```
torch>=2.0.0
transformers>=4.30.0
fire
numpy
```

### LLM Training
```
torch>=2.0.0
transformers>=4.30.0
fire
numpy
scikit-learn
```

## Hardware Requirements

- **4 GPUs** (configured for distributed training)
- **Memory**: Depends on model size
  - DeBERTa-base: ~8GB per GPU
  - Qwen2.5-0.5B: ~4GB per GPU
  - Larger models may require more memory

## Data Preparation

### DeBERTa Data Format
Each line should contain:
- `question`: The question text
- `llm_id`: Identifier for the LLM that should handle this question
- `label`: Binary label (0/1) for classification

### LLM Data Format
Standard Alpaca format:
- `instruction`: The question or task
- `response`: The expected response for training

## Output

### DeBERTa Training
- Trained model saved to `{output_dir}/`
- Checkpoints saved per epoch
- Model can be loaded with `DebertaV2ForSequenceClassification`

### LLM Training
- Trained model saved to `{output_dir}/`
- Compatible with standard transformers loading
- Model can be used for difficulty assessment

## Usage in Evaluation Pipeline

After training, the models are automatically integrated into the CoBench evaluation pipeline:

```python
# The pipeline will automatically use your trained router
python run.py --router_type trained_deberta --router_model_path your_model_path
```

## Examples

See `config_router_examples.yaml` for complete configuration examples.

## Notes

- All training scripts use `fire` for parameter management
- Distributed training is automatically handled via `torchrun`
- Models are saved in HuggingFace format for easy loading
- Training progress is logged to console