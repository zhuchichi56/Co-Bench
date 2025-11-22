

# import json
# import random

# # 读取原始数据
# data = []
# with open("../data/metamath_5k.jsonl", 'r', encoding='utf-8') as f:
#     for line in f:
#         data.append(json.loads(line))

# # 打乱数据（可选，确保随机分割）
# random.shuffle(data)

# # 分割数据
# train_data = data[:4000]
# test_data = data[4000:5000]

# # 保存训练集
# with open("../data/metamath_5k_train.jsonl", 'w', encoding='utf-8') as f:
#     for item in train_data:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')

# # 保存测试集
# with open("../data/metamath_5k_test.jsonl", 'w', encoding='utf-8') as f:
#     for item in test_data:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')

# print(f"数据分割完成！")
# print(f"训练集: {len(train_data)} 条 -> ../data/metamath_5k_train.jsonl")
# print(f"测试集: {len(test_data)} 条 -> ../data/metamath_5k_test.jsonl")

import json

def convert_dapo_to_simple_format(input_jsonl, output_jsonl):
    """
    将 DAPO 格式转换为简单格式
    
    输入格式:
    {
        "data_source": "math_dapo",
        "prompt": [{"content": "...", "role": "user"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": "34", "style": "..."},
        "extra_info": {"index": "..."}
    }
    
    输出格式:
    {
        "instruction": "...",
        "response": "34",
        "level": "MATH",
        "domain": "math_dapo"
    }
    """
    converted_count = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_jsonl, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            try:
                # 解析原始数据
                data = json.loads(line.strip())
                
                # 提取 instruction（从 prompt 中获取）
                if 'prompt' in data and len(data['prompt']) > 0:
                    instruction = data['prompt'][0].get('content', '')
                else:
                    instruction = ''
                
                # 提取 response（从 reward_model.ground_truth 获取）
                response = data.get('reward_model', {}).get('ground_truth', '')
                
                # 提取 level（使用 ability）
                level = data.get('ability', '')
                
                # 提取 domain（使用 data_source）
                domain = data.get('data_source', '')
                
                # 构建新格式
                new_data = {
                    'instruction': instruction,
                    'response': response,
                    'level': level,
                    'domain': domain
                }
                
                # 写入输出文件
                fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except Exception as e:
                print(f"警告：第 {line_num} 行转换失败: {e}")
                continue
    
    print(f"转换完成！共成功转换 {converted_count} 条数据")
    return converted_count


# 使用示例
if __name__ == '__main__':
    input_file = '../dapo-math-17k_dedup.jsonl'
    output_file = '../data/dapo-math-17k_dedup.jsonl'
    
    convert_dapo_to_simple_format(input_file, output_file)
    
    # 查看前几行
    print("\n前3行预览:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            print(f"\n[{i+1}]")
            print(f"  instruction: {data['instruction'][:80]}...")
            print(f"  response: {data['response']}")
            print(f"  level: {data['level']}")
            print(f"  domain: {data['domain']}")