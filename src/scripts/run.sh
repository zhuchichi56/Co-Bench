#!/bin/bash

# 激活conda环境
source /volume/pt-train/users/wzhang/ghchen/zh/miniconda3/bin/activate router

# 默认参数
DATASETS="${1:- hotpotqa_500}"
PROBE_TYPES="${2:-hs_last_mlp}"
MAX_SAMPLES="${3:-500}"


# MMLU Pro 测试数据集
MMLU_PRO_TASKS="mmlu_pro_biology mmlu_pro_business mmlu_pro_chemistry mmlu_pro_computer_science mmlu_pro_economics mmlu_pro_engineering mmlu_pro_health mmlu_pro_history mmlu_pro_law mmlu_pro_math mmlu_pro_other mmlu_pro_philosophy mmlu_pro_physics mmlu_pro_psychology"

# 完整测试数据集列表
TEST_DATASETS="${4:-math mmlu_pro_biology mmlu_pro_business mmlu_pro_chemistry mmlu_pro_computer_science mmlu_pro_economics mmlu_pro_engineering mmlu_pro_health mmlu_pro_history mmlu_pro_law mmlu_pro_math mmlu_pro_other mmlu_pro_philosophy mmlu_pro_physics mmlu_pro_psychology magpie_5k_test alpaca_5k_test big_math_5k_test mmlu_test}"
echo "========================================="
echo "CoBench 完整 Pipeline"
echo "========================================="
echo "训练数据集: $DATASETS"
echo "测试数据集: $TEST_DATASETS"
echo "Probe 类型: $PROBE_TYPES"
echo "最大样本数: $MAX_SAMPLES"

# ========================================
# 测试数据的步骤
# ========================================
# 启动模型服务
# cd inference
# vllm serve /volume/pt-train/users/wzhang/ghchen/zh/models/Qwen3-8B --chat-template ./qwen3_nonthinking.jinja \
#  python start.py \
#   --model_path  "/volume/pt-train/models/Llama-3.1-8B-Instruct" \
#   --base_port 8001 \
#   --gpu_list "6,7"



# # 如果测试非general数据 需要启动xVerify
# CUDA_VISIBLE_DEVICES=3 \
# vllm serve /volume/pt-train/users/wzhang/ghchen/zh/models/xVerify-9B-C \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --served-model-name xVerify \
#   --trust-remote-code

# # 等待模型服务启动完成后，运行 scores
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

# python run_new.py --mode get_scores --datasets $DATASETS
# # # logits
# python run_new.py --mode get_logits --datasets $DATASETS
# # training probe
# python run_new.py --mode train --datasets $DATASETS --probe_types $PROBE_TYPES --max_samples $MAX_SAMPLES
# python run_new.py --mode train --datasets $DATASETS



# export TRANSFORMERS_CACHE=/volume/pt-train/users/wzhang/ghchen/zh/models/longformer-base-4096
# export HF_HOME=/volume/pt-train/users/wzhang/ghchen/zh/models
# export HF_HUB_OFFLINE=1   # 可选，完全离线
# python run_new.py --mode get_query_emb --datasets $DATASETS
# python run_new.py --mode train --max_samples 4000 --datasets $DATASETS

# # 评估
# python run_new.py --mode eval_probe --datasets $TEST_DATASETS --probe_types $PROBE_TYPES
# python run_new.py --mode logits_based_routers --datasets $TEST_DATASETS 
# python run_new.py --mode self_based --datasets $TEST_DATASETS 
python run_new.py --mode eval_embedding_mlp  --datasets $DATASETS
 
# python run_new.py --mode eval_deberta --datasets hotpotqa_500




# ========================================
# 启动模型服务
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
