#!/usr/bin/env python3
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict
from tqdm import tqdm


def generate_logits_for_models(model_paths: List[str], dataset_path: str, output_dir: str = "logits_output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data_list = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    print(f"Processing {len(model_paths)} models with {len(data_list)} samples")

    for model_path in tqdm(model_paths, desc="Models"):
        print(f"Loading model from {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_device = next(model.parameters()).device
        results = []

        for item in tqdm(data_list, desc=f"Samples ({Path(model_path).name})"):
            instruction = item.get("instruction", "")
            score = item.get("score", 0.0)

            inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                all_hidden_states = []

                for layer_states in outputs.hidden_states:
                    layer_states = layer_states.cpu().numpy()
                    all_hidden_states.append(layer_states.squeeze(0))

                hidden_states = np.array(all_hidden_states)

            results.append((hidden_states.tolist(), score))

        model_name = Path(model_path).name
        task_name = Path(dataset_path).stem
        output_path = output_dir / f"{model_name}_{task_name}.pt"

        torch.save(results, output_path)
        print(f"Saved {len(results)} samples to {output_path}")

        del model, tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    model_paths = [
        "/volume/pt-train/models/Qwen2.5-7B-Instruct",
        "/volume/pt-train/models/Llama-3.1-8B-Instruct"
    ]

    dataset_path = "data/mmlu_subset.jsonl"

    generate_logits_for_models(
        model_paths=model_paths,
        dataset_path=dataset_path,
        output_dir="logits_output"
    )