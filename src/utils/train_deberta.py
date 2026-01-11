#!/usr/bin/env python3
"""
DeBERTa Training Program for Question Classification
Input format: question, label
Output format: Question -> label
Supports multi-GPU training (4 GPUs)
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    # AdamW,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
try:
    import fire  # type: ignore
except ModuleNotFoundError:
    fire = None
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Special separator token
SEP_TOKEN = "<SEP>"

class QuestionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict]:
        """Load data from jsonl file"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if all(key in item for key in ['question', 'label']):
                        data.append(item)
                except json.JSONDecodeError:
                    continue
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format input: Question only
        text = f"{item['question']}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

class DeBERTaTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer and model
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(config.model_name)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )

        # Add special token if not exists
        if SEP_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([SEP_TOKEN])
            self.model.resize_token_embeddings(len(self.tokenizer))

    def setup_distributed(self):
        """Setup distributed training"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])

            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')

            self.model = self.model.to(self.local_rank)
            self.model = DDP(self.model, device_ids=[self.local_rank])

            logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")
        else:
            self.model = self.model.to(self.device)
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

    def create_data_loader(self, dataset, batch_size, shuffle=True):
        """Create data loader with distributed sampling if needed"""
        if self.world_size > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

    def train(self, train_dataset, val_dataset=None):
        """Training loop"""
        train_loader = self.create_data_loader(
            train_dataset,
            self.config.batch_size
        )

        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = len(train_loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        self.model.train()
        for epoch in range(self.config.epochs):
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            epoch_loss = 0
            for step, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Log progress
                if step % 100 == 0 and self.rank == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs}, "
                              f"Step {step}/{len(train_loader)}, "
                              f"Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(train_loader)
            if self.rank == 0:
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

                # Save checkpoint
                self.save_model(f"checkpoint_epoch_{epoch+1}")

    def save_model(self, checkpoint_name):
        """Save model checkpoint"""
        if self.rank == 0:
            output_dir = Path(self.config.output_dir) / checkpoint_name
            output_dir.mkdir(parents=True, exist_ok=True)

            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            logger.info(f"Model saved to {output_dir}")

class TrainingConfig:
    def __init__(self):
        self.model_name = "microsoft/deberta-v3-base"
        self.num_labels = 2  # Binary classification, adjust as needed
        self.max_length = 512
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.epochs = 3
        self.output_dir = "deberta_checkpoints"

def main(
    data_path: str = "data/sample_data.jsonl",
    val_data_path: str = None,
    model_name: str = "microsoft/deberta-v3-base",
    num_labels: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    output_dir: str = "deberta_checkpoints",
    **kwargs
):
    """Train DeBERTa for question classification using fire"""

    # Create config
    config = TrainingConfig()
    config.model_name = model_name
    config.num_labels = num_labels
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.epochs = epochs
    config.output_dir = output_dir

    # Initialize trainer
    trainer = DeBERTaTrainer(config)
    trainer.setup_distributed()

    # Load datasets
    train_dataset = QuestionDataset(data_path, trainer.tokenizer)
    val_dataset = QuestionDataset(val_data_path, trainer.tokenizer) if val_data_path else None

    if trainer.rank == 0:
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Start training
    trainer.train(train_dataset, val_dataset)

    # Cleanup distributed training
    if trainer.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    if fire is None:
        raise SystemExit("Missing optional dependency 'fire'. Install with: pip install fire")
    fire.Fire(main)
