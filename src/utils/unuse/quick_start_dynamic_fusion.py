#!/usr/bin/env python3
"""
快速开始：使用训练好的动态融合模型作为CoBench Router

使用方法:
1. 确保你已经用 src_easy/test_dynamic.py 训练了模型
2. 修改下面的模型路径
3. 运行此脚本测试集成
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def quick_start():
    """快速开始使用动态融合router"""

    print("🚀 CoBench 动态融合Router - 快速开始")
    print("=" * 50)

    # 你的模型路径 - 请根据实际情况修改
    model_paths = {
        "softmax": "../probe_save_dynamic/mixed_magpie_5k_train_mmlu_train_numina_cot_5k_train_softmax_probe.pt",
        "dirichlet": "../probe_save_dynamic/mixed_magpie_5k_train_mmlu_train_numina_cot_5k_train_dirichlet_probe.pt"
    }

    print("📁 检查模型文件...")
    existing_models = {}
    for model_type, path in model_paths.items():
        if os.path.exists(path):
            print(f"  ✅ 找到 {model_type} 模型: {path}")
            existing_models[model_type] = path
        else:
            print(f"  ❌ 未找到 {model_type} 模型: {path}")

    if not existing_models:
        print("\n❌ 未找到任何训练好的模型!")
        print("请先运行 src_easy/test_dynamic.py 训练模型")
        return

    print(f"\n✅ 找到 {len(existing_models)} 个模型文件")

    # 现在展示如何在实际评估中使用
    usage_example = f"""
📖 实际使用示例:

# ========================================
# 在你的评估脚本中添加以下代码:
# ========================================

from router import create_router_manager

# 1. 创建router管理器
manager = create_router_manager()

# 2. 注册动态融合routers
"""

    for model_type, path in existing_models.items():
        usage_example += f'''
# 注册 {model_type} 动态融合router
manager.create_dynamic_fusion_router(
    checkpoint_path="{path}",
    probe_type="{model_type}",
    name="dynamic_fusion_{model_type}"
)'''

    usage_example += '''

# 3. 在评估循环中使用
def evaluate_with_dynamic_fusion(data, model_type="dirichlet"):
    router_name = f"dynamic_fusion_{model_type}"

    # 获取router评分 (0-1之间，越高表示越难/越需要强模型)
    scores = manager.get_router_scores(router_name, data)

    # 根据阈值决定路由
    threshold = 0.5
    for i, (item, score) in enumerate(zip(data, scores)):
        if score > threshold:
            print(f"样本 {i}: 路由到强模型 (难度: {score:.3f})")
            # 使用强模型处理
        else:
            print(f"样本 {i}: 路由到弱模型 (难度: {score:.3f})")
            # 使用弱模型处理

# 4. 不确定性分析 (仅限Dirichlet方法)
def analyze_uncertainty(data):
    # 直接使用DynamicFusionRouter获取不确定性
    from router import DynamicFusionRouter

    router = DynamicFusionRouter(
        checkpoint_path="''' + existing_models.get('dirichlet', 'path/to/dirichlet/model.pt') + '''",
        probe_type="dirichlet"
    )

    # 在这里可以添加不确定性分析的代码
    # router.model.forward(data, return_uncertainty=True)

# ========================================
# 替代现有router的最简方式:
# ========================================

# 如果你现在使用其他router，只需要改一行:
# OLD: router = ProbeRouter(checkpoint_path, "mean")
# NEW: router = ProbeRouter(checkpoint_path, "dynamic_dirichlet")

'''

    print(usage_example)

    # 展示可用的probe类型
    print("\n📋 现在CoBench支持的所有probe类型:")
    probe_types = [
        "hs_last_mlp", "coe_dual_mlp", "coe_c_scalar", "coe_r_scalar",
        "pca_conv", "mean", "max", "mean+max", "transformer",
        "dynamic_softmax",  # 新增
        "dynamic_dirichlet"  # 新增
    ]

    for i, pt in enumerate(probe_types, 1):
        marker = "🆕" if "dynamic" in pt else "  "
        print(f"  {marker} {i:2d}. {pt}")

    print("\n🎯 下一步:")
    print("  1. 将上述代码集成到你的评估脚本中")
    print("  2. 替换现有的router为dynamic_dirichlet")
    print("  3. 对比不同router的性能")
    print("  4. 分析Dirichlet方法的不确定性输出")

if __name__ == "__main__":
    quick_start()