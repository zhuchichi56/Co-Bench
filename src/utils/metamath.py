import os
import json
# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from collections import Counter
import random

# 设置随机种子
random.seed(54)

# 加载 MetaMath-40K 数据集
print("Loading MetaMath-40K dataset from hf-mirror...")
dataset = load_dataset("meta-math/MetaMathQA-40K")['train']

print(f"Total samples: {len(dataset)}")

# 统计各个 type 的数量
type_counts = Counter(dataset['type'])
print("\nType distribution:")
for type_name, count in sorted(type_counts.items()):
    percentage = count / len(dataset) * 100
    print(f"  {type_name}: {count} ({percentage:.2f}%)")

# 按 type 分层采样,总共 5000 条
sampled_indices = []
total_samples = len(dataset)
target_total = 5000

# 为每个 type 计算应该采样的数量
type_sample_counts = {}
for type_name, count in type_counts.items():
    # 按比例计算
    n_samples = int(target_total * count / total_samples)
    type_sample_counts[type_name] = n_samples

# 调整以确保总数正好是 5000
current_sum = sum(type_sample_counts.values())
if current_sum < target_total:
    # 如果不足,从最大的类别补充
    largest_type = max(type_counts, key=type_counts.get)
    type_sample_counts[largest_type] += (target_total - current_sum)

print(f"\nSampling strategy (total={sum(type_sample_counts.values())}):")
for type_name, n_samples in sorted(type_sample_counts.items()):
    print(f"  {type_name}: {n_samples}")

# 执行分层采样
for type_name, n_samples in type_sample_counts.items():
    # 找到该 type 的所有索引
    type_indices = [i for i, t in enumerate(dataset['type']) if t == type_name]
    
    # 随机采样
    if n_samples > 0:
        sampled = random.sample(type_indices, min(n_samples, len(type_indices)))
        sampled_indices.extend(sampled)
        print(f"Sampled {len(sampled)} from {type_name}")

# 创建采样后的数据集
print(f"\nTotal sampled: {len(sampled_indices)}")
sampled_dataset = dataset.select(sampled_indices)

# 验证采样后的 type 分布
sampled_type_counts = Counter(sampled_dataset['type'])
print("\nSampled type distribution:")
for type_name, count in sorted(sampled_type_counts.items()):
    percentage = count / len(sampled_dataset) * 100
    print(f"  {type_name}: {count} ({percentage:.2f}%)")

# 转换为 jsonl 格式,将 query 改为 instruction
output_path = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/data/metamath_5k_test.jsonl"

print(f"\nSaving to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as f:
    for item in sampled_dataset:
        # 创建新的字典,将 query 改为 instruction
        new_item = {
            "instruction": item['query'],
            "response": item['response'],
            "type": item['type']
        }
        # 写入一行 JSON
        f.write(json.dumps(new_item, ensure_ascii=False) + '\n')

print(f"Dataset saved to: {output_path}")
print(f"Total lines: {len(sampled_dataset)}")

# 打印前3个样本查看
print("\nFirst 3 samples:")
with open(output_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        data = json.loads(line)
        print(f"\n--- Sample {i+1} ---")
        print(f"Type: {data['type']}")
        print(f"Instruction: {data['instruction'][:100]}...")
        print(f"Response: {data['response'][:100]}...")