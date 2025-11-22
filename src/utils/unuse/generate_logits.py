#!/usr/bin/env python3
import torch
import json
from pathlib import Path
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple
import os

def load_model_and_tokenizer(model_path: str, device: str) -> Tuple:
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def extract_all_hidden_states(model, tokenizer, text: str, device: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        all_hidden_states = []

        for layer_states in outputs.hidden_states:
            layer_states = layer_states.cpu().numpy()
            all_hidden_states.append(layer_states.squeeze(0))

        return np.array(all_hidden_states)

def process_model_data(model_path: str, dataset_path: str, output_dir: str,
                      device_id: int = 0) -> str:
    device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu"

    # Set CUDA device for this process
    if torch.cuda.is_available() and device_id >= 0:
        torch.cuda.set_device(device_id)

    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # Load dataset
    data_list = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    results = []
    for item in data_list:
        instruction = item.get("instruction", "")
        score = item.get("score", 0.0)

        hidden_states = extract_all_hidden_states(model, tokenizer, instruction, device)
        results.append((hidden_states.tolist(), score))

    # Extract model name and task name
    model_name = Path(model_path).name
    task_name = Path(dataset_path).stem

    output_path = Path(output_dir) / f"{model_name}_{task_name}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(results, output_path)
    print(f"Saved {len(results)} samples to {output_path}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(output_path)

def process_model_wrapper(args: Tuple) -> str:
    model_path, dataset_path, output_dir, device_id = args
    return process_model_data(model_path, dataset_path, output_dir, device_id)

def generate_logits_for_models(model_paths: List[str],
                               dataset_path: str,
                               output_dir: str = "logits_output",
                               num_processes: int = None) -> List[str]:

    if num_processes is None:
        num_processes = min(len(model_paths), mp.cpu_count())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(model_paths)} models with dataset {dataset_path}")
    print(f"Using {num_processes} processes")

    process_args = []
    for i, model_path in enumerate(model_paths):
        device_id = i % torch.cuda.device_count() if torch.cuda.is_available() else -1
        process_args.append((model_path, dataset_path, output_dir, device_id))

    if num_processes == 1:
        output_paths = [process_model_wrapper(args) for args in process_args]
    else:
        # Use spawn method for CUDA compatibility
        ctx = mp.get_context('spawn')
        with ctx.Pool(num_processes) as pool:
            output_paths = pool.map(process_model_wrapper, process_args)

    print(f"Generated logits for all models. Output files: {output_paths}")
    return output_paths

# Example usage for testing
if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    model_paths = [
        "/path/to/llama-7b",
        "/path/to/mistral-7b",
        "/path/to/qwen-7b"
    ]

    dataset_path = "src/data/mmlu_test.jsonl"

    output_files = generate_logits_for_models(
        model_paths=model_paths,
        dataset_path=dataset_path,
        output_dir="logits_output"
    )
    # Output files will be: llama-7b_mmlu_test.pt, mistral-7b_mmlu_test.pt, qwen-7b_mmlu_test.pt