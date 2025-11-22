#!/usr/bin/env python3
"""
示例脚本：演示如何使用新的动态融合router
将训练好的模型集成到CoBench评估框架中
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from router import create_router_manager, DynamicFusionRouter
import torch
import numpy as np

def demonstrate_dynamic_fusion_router():
    """演示动态融合router的使用"""

    print("🚀 CoBench 动态融合Router集成示例")
    print("=" * 60)

    # 创建 RouterManager
    manager = create_router_manager()

    # 假设你有训练好的模型路径
    # 你需要将 src_easy 中训练的模型路径替换到这里
    softmax_model_path = "../probe_save_dynamic/mixed_magpie_5k_train_mmlu_train_numina_cot_5k_train_softmax_probe.pt"
    dirichlet_model_path = "../probe_save_dynamic/mixed_magpie_5k_train_mmlu_train_numina_cot_5k_train_dirichlet_probe.pt"

    print("📁 检查模型文件...")

    # 检查模型文件是否存在
    if os.path.exists(softmax_model_path):
        print(f"✅ 找到 Softmax 模型: {softmax_model_path}")

        # 注册 Softmax 动态融合router
        router_name_softmax = manager.create_dynamic_fusion_router(
            checkpoint_path=softmax_model_path,
            probe_type="softmax",
            name="dynamic_fusion_softmax"
        )
        print(f"✅ 注册 Softmax Router: {router_name_softmax}")

    else:
        print(f"❌ 未找到 Softmax 模型: {softmax_model_path}")

    if os.path.exists(dirichlet_model_path):
        print(f"✅ 找到 Dirichlet 模型: {dirichlet_model_path}")

        # 注册 Dirichlet 动态融合router
        router_name_dirichlet = manager.create_dynamic_fusion_router(
            checkpoint_path=dirichlet_model_path,
            probe_type="dirichlet",
            name="dynamic_fusion_dirichlet"
        )
        print(f"✅ 注册 Dirichlet Router: {router_name_dirichlet}")

    else:
        print(f"❌ 未找到 Dirichlet 模型: {dirichlet_model_path}")

    # 列出所有注册的router
    print(f"\n📋 已注册的Routers: {manager.list_routers()}")

    # 演示如何使用router进行评分
    print("\n🎯 演示Router评分功能...")

    # 创建一些模拟数据（实际使用时应该是真实的hidden states数据）
    mock_data = []
    for i in range(3):
        # 模拟 hidden states: [num_layers, hidden_dim]
        hidden_states = np.random.randn(32, 4096).astype(np.float32)
        mock_data.append({
            "hidden_states": hidden_states,
            "instruction": f"示例问题 {i+1}",
            "llm_id": "test_model"
        })

    # 测试每个router
    for router_name in manager.list_routers():
        if "dynamic_fusion" in router_name:
            try:
                scores = manager.get_router_scores(router_name, mock_data)
                print(f"  {router_name}: {scores}")
            except Exception as e:
                print(f"  {router_name}: 错误 - {e}")

def demonstrate_probe_router_integration():
    """演示通过ProbeRouter使用动态融合probe"""

    print("\n🔧 通过ProbeRouter使用动态融合...")

    # 也可以通过 ProbeRouter 直接使用
    from router import ProbeRouter

    softmax_model_path = "../probe_save_dynamic/mixed_magpie_5k_train_mmlu_train_numina_cot_5k_train_softmax_probe.pt"

    if os.path.exists(softmax_model_path):
        # 使用 dynamic_softmax probe type
        probe_router = ProbeRouter(
            checkpoint_path=softmax_model_path,
            probe_type="dynamic_softmax"
        )

        print("✅ 创建 ProbeRouter (dynamic_softmax)")

        # 创建模拟数据
        mock_data = []
        for i in range(2):
            hidden_states = np.random.randn(32, 4096).astype(np.float32)
            mock_data.append((hidden_states, 1.0))  # 元组格式

        scores = probe_router.get_router_scores(mock_data)
        print(f"ProbeRouter scores: {scores}")

    else:
        print("❌ 模型文件不存在，跳过ProbeRouter演示")

def show_integration_guide():
    """显示集成指南"""

    print("\n📖 集成指南")
    print("=" * 60)

    guide = """
🎯 如何在CoBench中使用动态融合Router:

1. 训练模型 (使用 src_easy/):
   cd src_easy
   python test_dynamic.py  # 这会训练并保存模型

2. 在评估脚本中使用:
   ```python
   from router import create_router_manager

   manager = create_router_manager()

   # 注册 Softmax 动态融合router
   manager.create_dynamic_fusion_router(
       checkpoint_path="path/to/softmax_model.pt",
       probe_type="softmax",
       name="my_softmax_router"
   )

   # 注册 Dirichlet 动态融合router
   manager.create_dynamic_fusion_router(
       checkpoint_path="path/to/dirichlet_model.pt",
       probe_type="dirichlet",
       name="my_dirichlet_router"
   )

   # 使用router进行评分
   scores = manager.get_router_scores("my_dirichlet_router", data)
   ```

3. 通过ProbeRouter使用:
   ```python
   from router import ProbeRouter

   # 使用dynamic_softmax或dynamic_dirichlet作为probe_type
   router = ProbeRouter(
       checkpoint_path="model.pt",
       probe_type="dynamic_dirichlet"  # 或 "dynamic_softmax"
   )
   ```

4. 可用的probe类型:
   - "dynamic_softmax": 原始softmax权重方法
   - "dynamic_dirichlet": Dirichlet分布建模方法

5. 数据格式:
   - 支持字典格式: {"hidden_states": np.array, ...}
   - 支持元组格式: (hidden_states_array, label)
   - hidden_states应该是[num_layers, hidden_dim]的numpy数组

🔥 优势:
   - Dirichlet方法提供不确定性量化
   - 自动学习层权重分布
   - 无缝集成到现有CoBench框架
   - 支持两种理论方法对比
"""

    print(guide)

if __name__ == "__main__":
    demonstrate_dynamic_fusion_router()
    demonstrate_probe_router_integration()
    show_integration_guide()