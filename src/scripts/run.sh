#!/bin/bash
set -euo pipefail

# Wrapper for split scripts:
# - prepare_all.sh
# - train_probes.sh
# - eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmd="${1:-}"
shift || true

# MMLU-Pro test datasets
MMLU_PRO_TASKS="mmlu_pro_biology mmlu_pro_business mmlu_pro_chemistry mmlu_pro_computer_science mmlu_pro_economics mmlu_pro_engineering mmlu_pro_health mmlu_pro_history mmlu_pro_law mmlu_pro_math mmlu_pro_other mmlu_pro_philosophy mmlu_pro_physics mmlu_pro_psychology"

# Full test dataset list
TEST_DATASETS="${4:-math mmlu_pro_biology mmlu_pro_business mmlu_pro_chemistry mmlu_pro_computer_science mmlu_pro_economics mmlu_pro_engineering mmlu_pro_health mmlu_pro_history mmlu_pro_law mmlu_pro_math mmlu_pro_other mmlu_pro_philosophy mmlu_pro_physics mmlu_pro_psychology magpie_5k_test alpaca_5k_test big_math_5k_test mmlu_test}"
echo "========================================="
echo "CoBench full pipeline"
echo "========================================="
echo "train_datasets: $DATASETS"
echo "test_datasets: $TEST_DATASETS"
echo "probe_types: $PROBE_TYPES"
echo "max_samples: $MAX_SAMPLES"

# ========================================
# Evaluation steps
# ========================================
# Start model server
cd inference
 python start.py \
  --model_path  "/volume/pt-train/models/Llama-3.1-8B-Instruct" \
  --base_port 8001 \
  --gpu_list "1,2"



# # If evaluating non-general datasets, start xVerify as well.
CUDA_VISIBLE_DEVICES=3 \
vllm serve /volume/pt-train/users/wzhang/ghchen/zh/models/xVerify-9B-C \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --served-model-name xVerify \
  --trust-remote-code

# # After the model server is ready, run scoring.
# # scores

# CUDA_VISIBLE_DEVICES=0 vllm serve /volume/pt-train/models/Qwen3-8B \
#     --host 0.0.0.0 \
#     --port 8001 \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.95 \
#     --enable_prefix_caching

# python agent.py\
#     --agent_name /volume/pt-train/models/Qwen3-8B \
#     --dataset med_qa_1k\
#     --agent_tools DuckDuckGoSearchTool \
#     --max_steps 5\     --concurrent_limit 20  \
#     --n_runs 1     --use_openai_server     --api_base "http://localhost:8001/v1"

# python main.py --mode prepare --datasets $DATASETS
# # # logits
# python main.py --mode prepare --datasets $DATASETS
# # training probe
# python main.py --mode train --datasets $DATASETS



# export TRANSFORMERS_CACHE=/volume/pt-train/users/wzhang/ghchen/zh/models/longformer-base-4096
# export HF_HOME=/volume/pt-train/users/wzhang/ghchen/zh/models
# export HF_HUB_OFFLINE=1   # optional: fully offline
# python main.py --mode prepare --datasets $DATASETS
# python main.py --mode train --datasets $DATASETS

# # Evaluation
# python main.py --mode eval --datasets $TEST_DATASETS
python main.py --mode eval --datasets $DATASETS
 
# python main.py --mode eval --datasets hotpotqa_500




# ========================================
# Start model server
# cd inference
# ts --gpu_indices 0,1,2,3 python start.py \
#   --model_path "/mnt/yixiali/MODELS/meta-llama/Llama-3.1-8B-Instruct" \
#   --base_port 8001 \
#   --gpu_list "0,1,2,3"


# ts -G 1 vllm serve IAAR-Shanghai/xVerify-9B-C \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --served-model-name xVerify \
#   --trust-remote-code



# conda activate;cd src
# ts bash run.sh alpaca_10k
# ts bash run.sh big_math_10k
