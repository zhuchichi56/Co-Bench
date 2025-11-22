---
license: mit
tags:
- pytorch
- probe
- dirichlet
---

# Dynamic Dirichlet Probe

Dynamic layer fusion probe using Dirichlet distribution.

## Parameters

- **sentence-level hidden_states**: `[batch_size, 29, 1536]` - Hidden states from each layer in Llama-3.1-8B
- **concentration_logits**: `[29]` - Per-layer concentration parameters
- **global_concentration**: `scalar` - Global concentration scale
- **classifier**: `Linear(1536 → 1)`

## Input
```python
hidden_states: [batch_size, 29, 1536]
```

## Usage
```python
import torch
import os
from huggingface_hub import hf_hub_download


model_path = hf_hub_download(repo_id="wanxwu/dynamic-dirichlet-probe-qwen2.5-numina", filename="pytorch_model.bin")
state_dict = torch.load(model_path)
```
