#!/bin/bash

# LLM SFT Training Script for 4 GPUs
# Usage: ./run_train_llm.sh <data_path> [model_path] [optional_args]

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters

DATA_PATH=${1:-"data/alpaca_5k.jsonl"}
MODEL_PATH=${2:-"/data1/wwx/models/models/Qwen2.5-0.5B"}

OUTPUT_DIR=${3:-""}
BATCH_SIZE=${4:-4}
GLOBAL_BATCH_SIZE=${5:-64}
LEARNING_RATE=${6:-2e-5}
EPOCHS=${7:-3}
MAX_LENGTH=${8:-512}

# Number of GPUs to use
NUM_GPUS=4

echo "Starting LLM SFT training with 4 GPUs..."
echo "Data path: $DATA_PATH"
echo "Model path: $MODEL_PATH"
echo "Per device batch size: $BATCH_SIZE"
echo "Global batch size: $GLOBAL_BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Max length: $MAX_LENGTH"

# Check if data file exists (relative to src directory)
if [ ! -f "$SRC_DIR/$DATA_PATH" ]; then
    echo "Error: Training data file not found: $SRC_DIR/$DATA_PATH"
    exit 1
fi

# Set output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="output/sft/${MODEL_NAME}"
fi

echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$SRC_DIR/$OUTPUT_DIR"

# Build fire arguments
FIRE_ARGS="--data_path=$DATA_PATH"
FIRE_ARGS="$FIRE_ARGS --model_name_or_path=$MODEL_PATH"
FIRE_ARGS="$FIRE_ARGS --output_dir=$OUTPUT_DIR"
FIRE_ARGS="$FIRE_ARGS --per_device_train_batch_size=$BATCH_SIZE"
FIRE_ARGS="$FIRE_ARGS --global_batch_size=$GLOBAL_BATCH_SIZE"
FIRE_ARGS="$FIRE_ARGS --learning_rate=$LEARNING_RATE"
FIRE_ARGS="$FIRE_ARGS --num_train_epochs=$EPOCHS"
FIRE_ARGS="$FIRE_ARGS --model_max_length=$MAX_LENGTH"

# Change to src directory
cd "$SRC_DIR"

# Run distributed training with torchrun
echo "Launching distributed training with torchrun..."
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_llm.py $FIRE_ARGS

echo "Training completed successfully!"
echo "Model saved to: $SRC_DIR/$OUTPUT_DIR"