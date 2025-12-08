#!/bin/bash

# 激活conda环境
# source /volume/pt-train/users/wzhang/ghchen/zh/miniconda3/bin/activate llf

# 默认参数
DATASETS="${1:- alpaca_5k_test}"
PROBE_TYPES="${2:- mean}" #hs_last_mlp mean max
MAX_SAMPLES="${3:-4000}"

echo "========================================="
echo "CoBench 完整 Pipeline"
echo "========================================="
echo "数据集: $DATASETS"
echo "Probe 类型: $PROBE_TYPES"
echo "最大样本数: $MAX_SAMPLES"

# ========================================
# 测试数据的步骤
# ========================================
# 启动模型服务
# cd inference
# python start.py \
#   --model_path "/volume/pt-train/models/Llama-3.1-8B-Instruct" \
#   --base_port 8001 \
#   --gpu_list "0,1,2,3"


# 如果测试非general数据 需要启动xVerify
# CUDA_VISIBLE_DEVICES=4 \
# vllm serve /volume/pt-train/users/wzhang/ghchen/zh/models/xVerify-9B-C \
#   --host 0.0.0.0 \
#   --port 8000 \
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
# python run.py --mode eval_probe --datasets $DATASETS --probe_types $PROBE_TYPES

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