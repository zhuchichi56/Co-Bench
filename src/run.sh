#!/bin/bash

# 激活conda环境
source /volume/pt-train/users/wzhang/ghchen/zh/miniconda3/bin/activate router

# 默认参数
DATASETS="${1:- mmlu_subset}"
PROBE_TYPES="${2:-dynamic_dirichlet}"
MAX_SAMPLES="${3:-8000}"


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
# vllm serve /volume/pt-train/users/wzhang/ghchen/zh/models/Qwen3-8B --chat-template ./qwen3_nonthinking.jinja
#  python start.py \
#   --model_path "/volume/pt-train/users/wzhang/ghchen/zh/models/Qwen3-8B" \
#   --base_port 8001 \
#   --gpu_list "4,5"



# 如果测试非general数据 需要启动xVerify
# CUDA_VISIBLE_DEVICES=7 \
# vllm serve /volume/pt-train/users/wzhang/ghchen/zh/models/xVerify-9B-C \
#   --host 0.0.0.0 \
#   --port 8002 \
#   --tensor-parallel-size 1 \
#   --served-model-name xVerify \
#   --trust-remote-code

# 等待模型服务启动完成后，运行 scores
# scores
python run_new.py --mode get_scores --datasets $DATASETS
# # # logits
# python run_new.py --mode get_logits --datasets $DATASETS
# # training probe
# python run_new.py --mode train --datasets $DATASETS --probe_types $PROBE_TYPES --max_samples $MAX_SAMPLES --save_loss_history


# # 评估
# python run_new.py --mode eval_probe --datasets $TEST_DATASETS --probe_types $PROBE_TYPES
# python run_new.py --mode logits_based_routers --datasets $TEST_DATASETS 
# python run_new.py --mode self_based --datasets $TEST_DATASETS 
echo ""
echo "========================================="
echo "🎉 完整 Pipeline 执行成功！"
echo "========================================="
echo "结果保存位置:"
echo "  - Scores:    results/"
echo "  - Logits:    ../hs/"
echo "  - 模型:      probe_save/test/"
echo "  - 训练历史:   probe_save/loss/"
echo "  - 评估结果:   metric_results/eval/"


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