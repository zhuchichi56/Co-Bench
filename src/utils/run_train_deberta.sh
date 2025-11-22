#!/bin/bash

# DeBERTa 4-GPU Training Script
# Usage: ./run_train_deberta.sh <data_path> [optional_args]

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters
DATA_PATH=${1:-"data/sample_data.jsonl"}
VAL_DATA_PATH=${2:-""}
MODEL_NAME=${3:-"microsoft/deberta-v3-base"}
NUM_LABELS=${4:-2}
BATCH_SIZE=${5:-16}
LEARNING_RATE=${6:-2e-5}
EPOCHS=${7:-3}
OUTPUT_DIR=${8:-"deberta_checkpoints"}

# Number of GPUs to use
NUM_GPUS=4

echo "Starting DeBERTa training with 4 GPUs..."
echo "Data path: $DATA_PATH"
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Output directory: $OUTPUT_DIR"

# Check if data file exists (relative to src directory)
if [ ! -f "$SRC_DIR/$DATA_PATH" ]; then
    echo "Error: Training data file not found: $SRC_DIR/$DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$SRC_DIR/$OUTPUT_DIR"

# Build fire arguments
FIRE_ARGS="--data_path=$DATA_PATH"
FIRE_ARGS="$FIRE_ARGS --model_name=$MODEL_NAME"
FIRE_ARGS="$FIRE_ARGS --num_labels=$NUM_LABELS"
FIRE_ARGS="$FIRE_ARGS --batch_size=$BATCH_SIZE"
FIRE_ARGS="$FIRE_ARGS --learning_rate=$LEARNING_RATE"
FIRE_ARGS="$FIRE_ARGS --epochs=$EPOCHS"
FIRE_ARGS="$FIRE_ARGS --output_dir=$OUTPUT_DIR"

if [ -n "$VAL_DATA_PATH" ] && [ -f "$SRC_DIR/$VAL_DATA_PATH" ]; then
    FIRE_ARGS="$FIRE_ARGS --val_data_path=$VAL_DATA_PATH"
fi

# Change to src directory
cd "$SRC_DIR"

# Run distributed training with torchrun
echo "Launching distributed training..."
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_deberta.py $FIRE_ARGS

echo "Training completed successfully!"
echo "Model checkpoints saved to: $SRC_DIR/$OUTPUT_DIR"