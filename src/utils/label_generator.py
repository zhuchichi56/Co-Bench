#!/usr/bin/env python3
"""
打标签代码：比较两个模型的结果文件，根据分数生成标签
小模型分数 > (大模型分数 + 小模型分数) / 2 时标签为1，否则为0
"""

import json
import os
from typing import Dict, List, Tuple

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def generate_labels(
    gpt4o_file: str, 
    small_model_file: str, 
    output_file: str
) -> None:
    """
    生成标签：比较GPT-4o和小模型的分数
    
    Args:
        gpt4o_file: GPT-4o结果文件路径
        small_model_file: 小模型结果文件路径  
        output_file: 输出文件路径
    """
    
    print(f"正在加载GPT-4o结果文件: {gpt4o_file}")
    gpt4o_data = load_jsonl(gpt4o_file)
    
    print(f"正在加载小模型结果文件: {small_model_file}")
    small_model_data = load_jsonl(small_model_file)
    
    print(f"GPT-4o数据条数: {len(gpt4o_data)}")
    print(f"小模型数据条数: {len(small_model_data)}")
    
    # 创建小模型数据的索引，方便查找
    small_model_index = {}
    for item in small_model_data:
        # 根据instruction字段匹配
        instruction = item.get('instruction', '')
        small_model_index[instruction] = item
    
    labeled_data = []
    matched_count = 0
    unmatched_count = 0
    
    for i, gpt4o_item in enumerate(gpt4o_data):
        # 获取GPT-4o的instruction
        gpt4o_instruction = gpt4o_item.get('instruction', '')
        
        # 查找对应的小模型数据
        if gpt4o_instruction in small_model_index:
            small_model_item = small_model_index[gpt4o_instruction]
            
            # 获取分数
            gpt4o_score = gpt4o_item.get('score', 0.0)
            small_model_score = small_model_item.get('score', 0.0)
            
            # 计算平均分数
            avg_score = (gpt4o_score + small_model_score) / 2
            
            # 生成标签：小模型分数 > 平均分数时为1，否则为0
            label = 1 if small_model_score >= gpt4o_score else 0
            
            # 创建标签数据
            labeled_item = {
                'question_id': gpt4o_item.get('question_id', i + 1),
                'instruction': gpt4o_instruction,
                'gpt4o_score': gpt4o_score,
                'small_model_score': small_model_score,
                'average_score': avg_score,
                'label': label,
                'large_response': gpt4o_item.get('choices', [{}])[0].get('turns', [''])[0] if 'choices' in gpt4o_item else '',
                'generated_response': small_model_item.get('small_response', '')
            }
            
            labeled_data.append(labeled_item)
            matched_count += 1
            
            if matched_count <= 5:  # 打印前5个样例
                print(f"样例 {matched_count}:")
                print(f"  Instruction: {gpt4o_instruction[:100]}...")
                print(f"  GPT-4o分数: {gpt4o_score}")
                print(f"  小模型分数: {small_model_score}")
                print(f"  平均分数: {avg_score:.3f}")
                print(f"  标签: {label}")
                print()
        else:
            unmatched_count += 1
            print(f"警告: 未找到匹配的小模型数据 for instruction: {gpt4o_instruction[:50]}...")
    
    print(f"匹配成功: {matched_count} 条")
    print(f"未匹配: {unmatched_count} 条")
    
    # 统计标签分布
    if labeled_data:
        label_1_count = sum(1 for item in labeled_data if item['label'] == 1)
        label_0_count = len(labeled_data) - label_1_count
        
        print(f"标签分布:")
        print(f"  标签1 (小模型更好): {label_1_count} 条 ({label_1_count/len(labeled_data)*100:.1f}%)")
        print(f"  标签0 (小模型较差): {label_0_count} 条 ({label_0_count/len(labeled_data)*100:.1f}%)")
    else:
        print("没有生成任何标签数据")
    
    # 保存结果
    print(f"正在保存结果到: {output_file}")
    save_jsonl(labeled_data, output_file)
    print("标签生成完成！")

def main():
    """主函数"""
    # 文件路径
    gpt4o_file = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/results/gpt-4o/mt-bench.jsonl"
    small_model_file = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/results/Llama-3.1-8B-Instruct/mt-bench.jsonl"
    output_file = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/results/labeled_mt-bench.jsonl"
    
    # 检查文件是否存在
    if not os.path.exists(gpt4o_file):
        print(f"错误: GPT-4o文件不存在: {gpt4o_file}")
        return
    
    if not os.path.exists(small_model_file):
        print(f"错误: 小模型文件不存在: {small_model_file}")
        return
    
    # 生成标签
    generate_labels(gpt4o_file, small_model_file, output_file)

if __name__ == "__main__":
    main()
