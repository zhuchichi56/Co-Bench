"""
Big-Math difficulty-stratified sampling script.

Goal: construct a benchmark that creates a clear performance gap between a 7B model and GPT-4o.
"""

from datasets import load_dataset
import pandas as pd
from collections import Counter
import random

# Set random seed for reproducibility
random.seed(42)

def select_by_difficulty(ds, low, high, name=""):
    """Filter examples by solve_rate range."""
    filtered = ds.filter(
        lambda ex: low <= ex["llama8b_solve_rate"] < high
    )
    print(f"{name}: solve_rate [{low}, {high}) -> {len(filtered)} problems")
    return filtered

def stratified_sample(ds, n_samples, stratify_by="source"):
    """Stratified sampling to balance across source/domain."""
    df = ds.to_pandas()
    
    # Distribution by stratum key
    source_counts = Counter(df[stratify_by])
    print(f"\nOriginal distribution ({stratify_by}):")
    for src, cnt in source_counts.most_common(10):
        print(f"  {src}: {cnt}")
    
    # Allocate samples per stratum (proportional)
    total = len(df)
    samples_per_source = {
        src: max(1, int(n_samples * cnt / total)) 
        for src, cnt in source_counts.items()
    }
    
    # Adjust to match total exactly
    diff = n_samples - sum(samples_per_source.values())
    if diff > 0:
        # Add to largest strata
        largest_sources = [src for src, _ in source_counts.most_common(abs(diff))]
        for src in largest_sources:
            samples_per_source[src] += 1
    elif diff < 0:
        # Subtract from largest strata
        largest_sources = [src for src, _ in source_counts.most_common(abs(diff))]
        for src in largest_sources:
            samples_per_source[src] = max(1, samples_per_source[src] - 1)
    
    # Sample within each stratum
    sampled_dfs = []
    for src, n in samples_per_source.items():
        src_df = df[df[stratify_by] == src]
        if len(src_df) >= n:
            sampled = src_df.sample(n=n, random_state=42)
        else:
            sampled = src_df  # take all if insufficient
        sampled_dfs.append(sampled)
    
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\nPost-sampling distribution ({stratify_by}):")
    sampled_counts = Counter(result_df[stratify_by])
    for src, cnt in sampled_counts.most_common(10):
        print(f"  {src}: {cnt}")
    
    return result_df

def main(total_samples=10000, train_ratio=0.8):
    """
    Args:
        total_samples: total number of samples to select (default: 10000)
        train_ratio: train split ratio (default: 0.8; 80% train / 20% test)
    """
    print("=" * 60)
    print("Loading Big-Math-RL-Verified-Processed dataset...")
    print("=" * 60)
    print(f"Target sample count: {total_samples}")
    print(f"Train/Test split ratio: {train_ratio:.0%} / {1-train_ratio:.0%}")

    # Output directory
    output_dir = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/data"
    
    # Load full dataset
    ds = load_dataset(
        "open-r1/Big-Math-RL-Verified-Processed",
        "all", 
        split="train"
    )
    
    print(f"\nTotal problems: {len(ds)}")
    print(f"Fields: {ds.features.keys()}")
    
    # solve_rate stats
    solve_rates = [ex["llama8b_solve_rate"] for ex in ds]
    print(f"\nsolve_rate stats:")
    print(f"  min: {min(solve_rates):.3f}")
    print(f"  max: {max(solve_rates):.3f}")
    print(f"  mean: {sum(solve_rates)/len(solve_rates):.3f}")
    
    # Filter out MATH source
    print("\n" + "=" * 60)
    print("Filtering out MATH source...")
    print("=" * 60)
    original_count = len(ds)
    ds = ds.filter(lambda ex: ex.get("source", "").lower() != "math")
    print(f"Before filter: {original_count} problems")
    print(f"After filter: {len(ds)} problems")
    print(f"Removed: {original_count - len(ds)} problems")
    
    # Remaining source distribution
    remaining_sources = Counter([ex.get("source", "unknown") for ex in ds])
    print(f"\nRemaining source distribution:")
    for src, cnt in remaining_sources.most_common(10):
        print(f"  {src}: {cnt}")
    
    print("\n" + "=" * 60)
    print("Starting difficulty-stratified sampling...")
    print("=" * 60)

    # Difficulty tier ratios (keep original proportions).
    # Target: ~50% pass rate for Llama-3.1-8B and ~75% for GPT-4o.
    # Original (10k total):
    # - easy (0.8-1.0): 30%
    # - medium_easy (0.6-0.8): 30%
    # - medium_hard (0.4-0.6): 20%
    # - hard (0.2-0.4): 14%
    # - very_hard (<0.2): 1.5%
    # Note: remaining 4.5% is used to adjust counts to match total exactly.

    difficulty_ratios = {
        "easy": 0.30,           # easy
        "medium_easy": 0.30,    # medium (easier)
        "medium_hard": 0.20,    # medium (harder)
        "hard": 0.14,           # hard
        "very_hard": 0.015      # very hard
    }

    # Compute target sample count per tier
    n_easy = int(total_samples * difficulty_ratios["easy"])
    n_medium_easy = int(total_samples * difficulty_ratios["medium_easy"])
    n_medium_hard = int(total_samples * difficulty_ratios["medium_hard"])
    n_hard = int(total_samples * difficulty_ratios["hard"])
    n_very_hard = int(total_samples * difficulty_ratios["very_hard"])

    # Adjust to match total exactly
    current_total = n_easy + n_medium_easy + n_medium_hard + n_hard + n_very_hard
    diff = total_samples - current_total
    if diff > 0:
        # Add to the largest tier (easy)
        n_easy += diff
    elif diff < 0:
        # Subtract from the largest tier (easy)
        n_easy += diff  # diff is negative

    print(f"\nDifficulty tier allocation (total {total_samples} problems):")
    print(f"  - easy (0.8-1.0): {n_easy} ({n_easy/total_samples*100:.1f}%)")
    print(f"  - medium_easy (0.6-0.8): {n_medium_easy} ({n_medium_easy/total_samples*100:.1f}%)")
    print(f"  - medium_hard (0.4-0.6): {n_medium_hard} ({n_medium_hard/total_samples*100:.1f}%)")
    print(f"  - hard (0.2-0.4): {n_hard} ({n_hard/total_samples*100:.1f}%)")
    print(f"  - very_hard (<0.2): {n_very_hard} ({n_very_hard/total_samples*100:.1f}%)")
    
    print("\n1. Select easy (solve_rate: 0.8-1.0)")
    easy = select_by_difficulty(ds, 0.8, 1.01, "easy")
    
    print("\n2. Select medium_easy (solve_rate: 0.6-0.8)")
    medium_easy = select_by_difficulty(ds, 0.6, 0.8, "medium_easy")
    
    print("\n3. Select medium_hard (solve_rate: 0.4-0.6)")
    medium_hard = select_by_difficulty(ds, 0.4, 0.6, "medium_hard")
    
    print("\n4. Select hard (solve_rate: 0.2-0.4)")
    hard = select_by_difficulty(ds, 0.2, 0.4, "hard")
    
    print("\n5. Select very_hard (solve_rate: 0.0-0.2)")
    very_hard = select_by_difficulty(ds, 0.0, 0.2, "very_hard")
    
    # Stratified sampling
    print("\n" + "=" * 60)
    print("Running stratified sampling (balanced by source)...")
    print("=" * 60)

    print(f"\n--- easy tier (target: {n_easy}) ---")
    easy_sampled = stratified_sample(easy, n_easy, "source")

    print(f"\n--- medium_easy tier (target: {n_medium_easy}) ---")
    medium_easy_sampled = stratified_sample(medium_easy, n_medium_easy, "source")

    print(f"\n--- medium_hard tier (target: {n_medium_hard}) ---")
    medium_hard_sampled = stratified_sample(medium_hard, n_medium_hard, "source")

    print(f"\n--- hard tier (target: {n_hard}) ---")
    hard_sampled = stratified_sample(hard, n_hard, "source")

    print(f"\n--- very_hard tier (target: {n_very_hard}) ---")
    very_hard_sampled = stratified_sample(very_hard, n_very_hard, "source")
    
    # Merge all tiers
    final_df = pd.concat([
        easy_sampled.assign(difficulty_tier="easy"),
        medium_easy_sampled.assign(difficulty_tier="medium_easy"),
        medium_hard_sampled.assign(difficulty_tier="medium_hard"),
        hard_sampled.assign(difficulty_tier="hard"),
        very_hard_sampled.assign(difficulty_tier="very_hard")
    ], ignore_index=True)
    
    # Rename: problem -> instruction, solution -> response
    # Inspect original columns
    print(f"\nOriginal columns: {list(final_df.columns)}")
    
    # Create a new DataFrame with only required fields
    output_df = pd.DataFrame()
    
    # Map Big-Math dataset fields:
    # - use 'problem' or 'question' as instruction
    # - use 'solution' or 'answer' as response
    if 'problem' in final_df.columns:
        output_df['instruction'] = final_df['problem']
    elif 'question' in final_df.columns:
        output_df['instruction'] = final_df['question']
    else:
        # If missing, print available columns to help debugging
        print("Warning: missing 'problem' and 'question' fields")
        print(f"Available columns: {list(final_df.columns)}")
        # Fallback: first text column
        text_cols = [col for col in final_df.columns if final_df[col].dtype == 'object']
        if text_cols:
            output_df['instruction'] = final_df[text_cols[0]]
    
    if 'solution' in final_df.columns:
        output_df['response'] = final_df['solution']
    elif 'answer' in final_df.columns:
        output_df['response'] = final_df['answer']
    else:
        print("Warning: missing 'solution' and 'answer' fields")
        # Fallback: second text column
        text_cols = [col for col in final_df.columns if final_df[col].dtype == 'object']
        if len(text_cols) > 1:
            output_df['response'] = final_df[text_cols[1]]
    
    # Keep metadata fields
    if 'llama8b_solve_rate' in final_df.columns:
        output_df['solve_rate'] = final_df['llama8b_solve_rate']
    output_df['difficulty_tier'] = final_df['difficulty_tier']
    if 'source' in final_df.columns:
        output_df['source'] = final_df['source']
    if 'domain' in final_df.columns:
        output_df['domain'] = final_df['domain']
    
    print("\n" + "=" * 60)
    print("Final benchmark stats")
    print("=" * 60)
    print(f"Total problems: {len(output_df)}")
    print(f"\nDifficulty tier distribution:")
    print(output_df['difficulty_tier'].value_counts())
    
    if 'source' in output_df.columns:
        print(f"\nSource distribution:")
        print(output_df['source'].value_counts())
    
    if 'domain' in output_df.columns:
        print(f"\nDomain distribution:")
        print(output_df['domain'].value_counts())
    
    # Save outputs
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Split train/test within each tier
    print("\n" + "=" * 60)
    print("Splitting Train/Test...")
    print("=" * 60)
    
    train_dfs = []
    test_dfs = []
    
    for tier in output_df['difficulty_tier'].unique():
        tier_df = output_df[output_df['difficulty_tier'] == tier]
        
        # Shuffle
        tier_df = tier_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split point (based on train_ratio)
        n_total = len(tier_df)
        n_train = int(n_total * train_ratio)
        
        tier_train = tier_df.iloc[:n_train]
        tier_test = tier_df.iloc[n_train:]
        
        train_dfs.append(tier_train)
        test_dfs.append(tier_test)
        
        print(f"{tier}: {n_total} -> train {len(tier_train)}, test {len(tier_test)}")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Shuffle final order
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal split:")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    
    print(f"\nTrain tier distribution:")
    print(train_df['difficulty_tier'].value_counts().sort_index())
    
    print(f"\nTest tier distribution:")
    print(test_df['difficulty_tier'].value_counts().sort_index())
    
    # Build filenames based on actual counts
    n_train_samples = len(train_df)
    n_test_samples = len(test_df)
    n_total_samples = len(output_df)

    # Save train
    train_file = os.path.join(output_dir, f"big_math_train_{n_train_samples//1000}k.jsonl")
    train_df.to_json(train_file, orient='records', lines=True, force_ascii=False)
    print(f"\n✓ Train saved to: {train_file}")

    # Save test
    test_file = os.path.join(output_dir, f"big_math_test_{n_test_samples//1000}k.jsonl")
    test_df.to_json(test_file, orient='records', lines=True, force_ascii=False)
    print(f"✓ Test saved to: {test_file}")

    # Optionally save full dataset
    output_file = os.path.join(output_dir, f"big_math_{n_total_samples//1000}k.jsonl")
    output_df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    print(f"✓ Full dataset saved to: {output_file}")
    
    # Write report
    report_file = os.path.join(output_dir, "big_math_sampling_report.txt")
    with open(report_file, 'w') as f:
        f.write("Big-Math Benchmark Sampling Report (10k, Train/Test split)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Overall stats\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total problems: {len(output_df)}\n")
        f.write(f"Train: {len(train_df)} (80%)\n")
        f.write(f"Test: {len(test_df)} (20%)\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Tier distribution (full)\n")
        f.write("=" * 60 + "\n")
        f.write(str(output_df['difficulty_tier'].value_counts().sort_index()) + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Tier distribution (train)\n")
        f.write("=" * 60 + "\n")
        f.write(str(train_df['difficulty_tier'].value_counts().sort_index()) + "\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Tier distribution (test)\n")
        f.write("=" * 60 + "\n")
        f.write(str(test_df['difficulty_tier'].value_counts().sort_index()) + "\n\n")
        
        if 'source' in output_df.columns:
            f.write("=" * 60 + "\n")
            f.write("Source distribution (full)\n")
            f.write("=" * 60 + "\n")
            f.write(str(output_df['source'].value_counts()) + "\n\n")
        
        if 'domain' in output_df.columns:
            f.write("=" * 60 + "\n")
            f.write("Domain distribution (full)\n")
            f.write("=" * 60 + "\n")
            f.write(str(output_df['domain'].value_counts()) + "\n\n")
        
        if 'solve_rate' in output_df.columns:
            f.write("=" * 60 + "\n")
            f.write("Solve rate stats\n")
            f.write("=" * 60 + "\n")
            f.write(f"Full:\n")
            f.write(f"  min: {output_df['solve_rate'].min():.3f}\n")
            f.write(f"  max: {output_df['solve_rate'].max():.3f}\n")
            f.write(f"  mean: {output_df['solve_rate'].mean():.3f}\n")
            f.write(f"  median: {output_df['solve_rate'].median():.3f}\n\n")
            
            f.write(f"Train:\n")
            f.write(f"  min: {train_df['solve_rate'].min():.3f}\n")
            f.write(f"  max: {train_df['solve_rate'].max():.3f}\n")
            f.write(f"  mean: {train_df['solve_rate'].mean():.3f}\n")
            f.write(f"  median: {train_df['solve_rate'].median():.3f}\n\n")
            
            f.write(f"Test:\n")
            f.write(f"  min: {test_df['solve_rate'].min():.3f}\n")
            f.write(f"  max: {test_df['solve_rate'].max():.3f}\n")
            f.write(f"  mean: {test_df['solve_rate'].mean():.3f}\n")
            f.write(f"  median: {test_df['solve_rate'].median():.3f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Expected model performance\n")
        f.write("=" * 60 + "\n")
        f.write("Expected accuracy based on tier distribution:\n\n")
        f.write("Llama-3.1-8B:\n")
        f.write("  - easy (0.8-1.0): ~70%\n")
        f.write("  - medium_easy (0.6-0.8): ~50%\n")
        f.write("  - medium_hard (0.4-0.6): ~32%\n")
        f.write("  - hard (0.2-0.4): ~18%\n")
        f.write("  - very_hard (<0.2): ~8%\n")
        f.write("  - overall: ~45-50%\n\n")
        f.write("GPT-4o:\n")
        f.write("  - easy (0.8-1.0): ~90%\n")
        f.write("  - medium_easy (0.6-0.8): ~80%\n")
        f.write("  - medium_hard (0.4-0.6): ~65%\n")
        f.write("  - hard (0.2-0.4): ~45%\n")
        f.write("  - very_hard (<0.2): ~25%\n")
        f.write("  - overall: ~70-75%\n\n")
        f.write("Expected gap: ~25-30%\n")
    
    print(f"✓ Report saved to: {report_file}")
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Big-Math difficulty-stratified sampling script")
    parser.add_argument(
        "--total_samples",
        type=int,
        default=10000,
        help="Total samples to select (default: 10000)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8; 80%% train / 20%% test)"
    )

    args = parser.parse_args()

    # Validate args
    if args.total_samples <= 0:
        raise ValueError("total_samples must be > 0")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")

    main(total_samples=args.total_samples, train_ratio=args.train_ratio)