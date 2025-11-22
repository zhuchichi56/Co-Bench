#!/usr/bin/env python3
"""
Simplified LLM Training Script (SFT) for Qwen3-0.6B
Extracted from the enhanced training script, keeping only SFT functionality
Supports multi-GPU training with 4 cards
"""

import os
import copy
import json
import fire
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer

# Disable wandb
os.environ["WANDB_MODE"] = "offline"

IGNORE_INDEX = -100

def load_jsonl(file_path: str):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# Special tokens
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# Prompt template
PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-0.5B")

@dataclass
class DataArguments:
    data_path: str = field(default="alpaca_data.jsonl")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    output_dir: str = field(default="./output")
    per_device_train_batch_size: int = field(default=4)
    num_train_epochs: float = field(default=3.0)
    learning_rate: float = field(default=2e-5)

def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer, model):
    """Resize tokenizer and embeddings for special tokens"""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    """Tokenize a list of strings"""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer) -> Dict:
    """Preprocess the data by tokenizing and masking"""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning"""

    def __init__(self, data_path: str, tokenizer):
        super(SupervisedDataset, self).__init__()
        list_data_dict = load_jsonl(data_path)

        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        sources = [prompt_no_input.format_map(example) for example in list_data_dict]
        targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning"""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def show_first_example(data_path: str, tokenizer):
    """Show first training example with tokenization details"""
    print("\n" + "="*50)
    print("FIRST TRAINING EXAMPLE")
    print("="*50)

    data = load_jsonl(data_path)
    if not data:
        print("No data found")
        return

    example = data[0]
    instruction = example.get('instruction', '')
    response = example.get('response', '')

    print(f"Instruction: {instruction}")
    print(f"Response: {response}")

    # Format with prompt
    prompt = PROMPT_DICT["prompt_no_input"].format_map(example)
    full_text = prompt + response + tokenizer.eos_token

    print(f"\nFull prompt:\n{prompt}")
    print(f"Target text: {response}{tokenizer.eos_token}")

    # Tokenize
    tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True
    )
    input_ids = tokenized.input_ids[0]

    # Calculate instruction length for masking
    instruction_tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True
    )
    instruction_len = instruction_tokenized.input_ids[0].shape[0]

    print(f"\nTokenization:")
    print(f"Total tokens: {len(input_ids)}")
    print(f"Instruction tokens: {instruction_len}")
    print(f"Response tokens: {len(input_ids) - instruction_len}")

    # Show loss computation part
    labels = input_ids.clone()
    labels[:instruction_len] = IGNORE_INDEX

    print(f"\nLoss computation tokens (response part):")
    loss_tokens = input_ids[instruction_len:]
    decoded_loss_part = tokenizer.decode(loss_tokens, skip_special_tokens=False)
    print(f"Tokens for loss: {decoded_loss_part}")
    print(f"Token IDs: {loss_tokens.tolist()}")
    print("="*50 + "\n")

def train(
    model_name_or_path: str = "/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen2.5-0.5B",
    data_path: str = "alpaca_data.jsonl",
    cache_dir: str = None,
    model_max_length: int = 512,
    per_device_train_batch_size: int = 4,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    global_batch_size: int = 64,
    output_dir: str = None,
    gradient_checkpointing: bool = True,
    dataloader_num_workers: int = 4,
    **kwargs
):
    """Simple SFT training function"""

    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    data_args = DataArguments(data_path=data_path)

    if output_dir is None:
        model_name = os.path.basename(model_name_or_path)
        output_dir = f"./output/sft/{model_name}"

    # Calculate gradient accumulation steps for distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = global_batch_size // (per_device_train_batch_size * world_size)

    print(f"World size: {world_size}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Per device batch size: {per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=gradient_checkpointing,
        ddp_find_unused_parameters=False,
        bf16=True,  # Use bfloat16 for better performance
        tf32=True,  # Enable tf32 for faster training on Ampere GPUs
        **kwargs
    )

    # Load model
    print(f"Loading model from {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load tokenizer
    print(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    # Handle special tokens
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

    # For Qwen models, ensure proper tokenizer setup
    if "qwen" in model_args.model_name_or_path.lower():
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "<|endoftext|>"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")

    # Show first example for debugging
    show_first_example(data_args.data_path, tokenizer)

    # Create data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    print(f"Training dataset size: {len(data_module['train_dataset'])}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    fire.Fire(train)