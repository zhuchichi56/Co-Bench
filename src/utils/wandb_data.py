import re
import pandas as pd
import numpy as np

output_log_1 = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/wandb/run-20251018_022423-7c7wlnkm/files/output.log"
output_log_2 = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/wandb/run-20251017_003235-jp6ko4tv/files/output.log"

def extract_scores_from_log(log_file):
    all_entries = []
    
    with open(log_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            if 'DEBUG: Extracted score:' in line:
                matches = re.findall(r'[-+]?\d*\.?\d+', line.split('score:')[-1])
                if matches:
                    all_entries.append({
                        'line_no': line_no,
                        'type': 'scored',
                        'score': float(matches[0]),
                        'content': line.strip()
                    })
            elif 'DEBUG: Skipping question' in line and 'missing answer' in line:
                question_match = re.search(r'question\s+(\d+)', line)
                all_entries.append({
                    'line_no': line_no,
                    'type': 'skipped',
                    'score': -1.0,
                    'question_id': int(question_match.group(1)) if question_match else None,
                    'content': line.strip()
                })
    
    scores = [e['score'] for e in all_entries]
    return scores, all_entries

# 提取文件1的scores (1043条)
print(f"📖 读取文件1: {output_log_1}")
scores_1, entries_1 = extract_scores_from_log(output_log_1)
print(f"✅ 文件1: {len(scores_1)} 条记录")

# 提取文件2的scores (4000条)
print(f"\n📖 读取文件2: {output_log_2}")
scores_2, entries_2 = extract_scores_from_log(output_log_2)
print(f"✅ 文件2: {len(scores_2)} 条记录")

# 检查长度
if len(scores_1) != 1043:
    print(f"⚠️  警告: 文件1有 {len(scores_1)} 条,不是1043条")

if len(scores_2) < 1043:
    print(f"❌ 错误: 文件2只有 {len(scores_2)} 条,少于1043条")
    exit()

# 替换: 文件2的前(N-1043)条 + 文件1的全部1043条
n_keep_from_2 = len(scores_2) - 1043
scores_combined = scores_2[:n_keep_from_2] + scores_1

print(f"\n🔄 合并:")
print(f"  文件2的前 {n_keep_from_2} 条")
print(f"  + 文件1的全部 {len(scores_1)} 条")
print(f"  = 总共 {len(scores_combined)} 条")

# 保存
df_combined = pd.DataFrame({'score': scores_combined})
df_combined.to_csv('scores_combined.csv', index=False)
np.save('scores_combined.npy', scores_combined)

print(f"\n✅ 保存到:")
print(f"  - scores_combined.csv")
print(f"  - scores_combined.npy")

# 统计
valid_scores = [s for s in scores_combined if s != -1]
print(f"\n📊 合并后统计:")
print(f"  总记录: {len(scores_combined)}")
print(f"  有效分数: {len(valid_scores)}")
print(f"  Missing(-1): {len(scores_combined) - len(valid_scores)}")

if valid_scores:
    print(f"  均值: {np.mean(valid_scores):.4f}")
    print(f"  中位数: {np.median(valid_scores):.4f}")
    print(f"  范围: [{np.min(valid_scores):.1f}, {np.max(valid_scores):.1f}]")

print(f"\n📈 分数分布:")
print(f"  -1: {sum(1 for s in scores_combined if s == -1)}")
print(f"  0-3: {sum(1 for s in scores_combined if 0 <= s <= 3)}")
print(f"  4-6: {sum(1 for s in scores_combined if 4 <= s <= 6)}")
print(f"  7-10: {sum(1 for s in scores_combined if 7 <= s <= 10)}")

# 额外保存详细对比
print(f"\n📋 生成详细对比...")
comparison = pd.DataFrame({
    'index': range(len(scores_combined)),
    'score': scores_combined,
    'source': ['file2'] * n_keep_from_2 + ['file1'] * len(scores_1)
})
comparison.to_csv('scores_combined_detail.csv', index=False)
print(f"✅ 保存详细信息到 scores_combined_detail.csv")

gpt = pd.read_json("/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/results/gpt-4o/alpaca_5k_train.jsonl", lines=True)
gpt['score'] = scores_combined
gpt.to_json('gpt4o_alpaca_5k_train_with_scores.jsonl', lines=True, orient='records')