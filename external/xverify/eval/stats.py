import os
import json
import pandas as pd
from collections import defaultdict
import glob

def analyze_model_results():
    # Initialize result containers
    results = {}
    
    # Find all result files under final_outputs
    result_files = glob.glob('final_outputs/*.json')
    
    # Track all dataset names
    all_datasets = set()
    
    # Process each model file
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace('.json', '')
        print(f"Processing model: {model_name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize per-model stats
        model_results = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))
        
        # Consistency vs human_judge
        consistency_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'consistent': 0}))
        
        # Process each item
        for item in data:
            dataset = item.get('data_source', 'unknown')
            all_datasets.add(dataset)
            
            human_judgment = item.get('human_judge')
            
            # Accuracy per judge
            for judge_type in ['regex_judge', 'judge_wo_ref', 'human_judge', 'xverify_judge']:
                judgment = item.get(judge_type)
                if judgment is not None:  # only when a judgment exists
                    model_results[dataset][judge_type]['total'] += 1
                    if judgment is False or judgment in ('incorrect', 'Incorrect'):
                        pass
                        judge_res = 'incorrect'
                    elif judgment is True or judgment in ('correct', 'Correct'):
                        model_results[dataset][judge_type]['correct'] += 1
                        judge_res = 'correct'
                        model_results['all'][judge_type]['correct'] += 1
                    model_results['all'][judge_type]['total'] += 1
                    
                    
                    # Consistency with human_judge
                    if judge_res == human_judgment:
                        consistency_stats[dataset][judge_type]['consistent'] += 1
                        consistency_stats['all'][judge_type]['consistent'] += 1
                    consistency_stats[dataset][judge_type]['total'] += 1
                    consistency_stats['all'][judge_type]['total'] += 1
                        
        # Store results for this model
        results[model_name] = {
            'accuracy': model_results,
            'consistency': consistency_stats
        }
    
    # Ensure 'all' exists in dataset list
    all_datasets = sorted(list(all_datasets))
    if 'all' not in all_datasets:
        all_datasets.append('all')
    
    # Create separate DataFrames per judge type
    judge_types = ['human_judge', 'judge_wo_ref', 'xverify_judge', 'regex_judge']
    
    # DataFrames to be written to Excel
    dataframes = {}
    
    # Accuracy DataFrames
    for judge_type in judge_types:
        df_data = []
        for model_name, model_data in results.items():
            row = {'Model': model_name}
            for dataset in all_datasets:
                stats = model_data['accuracy'].get(dataset, {}).get(judge_type, {'total': 0, 'correct': 0})
                if stats['total'] > 0:
                    row[dataset] = stats['correct'] / stats['total']
                else:
                    row[dataset] = None
            df_data.append(row)
        dataframes[f'Accuracy_{judge_type}'] = pd.DataFrame(df_data)
    
    # Consistency DataFrames
    for judge_type in ['regex_judge', 'judge_wo_ref', 'xverify_judge']:
        df_data = []
        for model_name, model_data in results.items():
            row = {'Model': model_name}
            for dataset in all_datasets:
                stats = model_data['consistency'].get(dataset, {}).get(judge_type, {'total': 0, 'consistent': 0})
                if stats['total'] > 0:
                    row[dataset] = stats['consistent'] / stats['total']
                else:
                    row[dataset] = None
            df_data.append(row)
        dataframes[f'Con_{judge_type}_vs_human'] = pd.DataFrame(df_data)
    
    # Save all DataFrames to separate sheets in one Excel file
    with pd.ExcelWriter('model_evaluation_results.xlsx') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print("Done. Results saved to model_evaluation_results.xlsx")

if __name__ == "__main__":
    analyze_model_results()