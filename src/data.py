import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from loss_calculator import MultiModalLossCalculator
from inference.vllm_client import parallel_inference
from inference.gpt_inference import parallel_inference_gpt
from loss_calculator import evaluate_skywork_reward, llm_judge_general,llm_as_a_judge


class DatasetConfig:
    def __init__(self, name: str, type: str, file_path: str):
        self.name = name
        self.type = type  # "mmlu", "math", "general"
        self.file_path = file_path


class DatasetRegistry:
    DATASETS = {
        "mmlu": DatasetConfig("mmlu", "mmlu", "mmlu_subset.jsonl"),
        # "mmlupro": DatasetConfig("mmlupro", "mmlupro", "mmlupro.jsonl"),
        # MMLU Pro数据集
        "mmlu_pro_biology": DatasetConfig("mmlu_pro_biology", "mmlu", "mmlu_pro/biology_converted.jsonl"),
        "mmlu_pro_business": DatasetConfig("mmlu_pro_business", "mmlu", "mmlu_pro/business_converted.jsonl"),
        "mmlu_pro_chemistry": DatasetConfig("mmlu_pro_chemistry", "mmlu", "mmlu_pro/chemistry_converted.jsonl"),
        "mmlu_pro_computer_science": DatasetConfig("mmlu_pro_computer_science", "mmlu", "mmlu_pro/computer_science_converted.jsonl"),
        "mmlu_pro_economics": DatasetConfig("mmlu_pro_economics", "mmlu", "mmlu_pro/economics_converted.jsonl"),
        "mmlu_pro_engineering": DatasetConfig("mmlu_pro_engineering", "mmlu", "mmlu_pro/engineering_converted.jsonl"),
        "mmlu_pro_health": DatasetConfig("mmlu_pro_health", "mmlu", "mmlu_pro/health_converted.jsonl"),
        "mmlu_pro_history": DatasetConfig("mmlu_pro_history", "mmlu", "mmlu_pro/history_converted.jsonl"),
        "mmlu_pro_law": DatasetConfig("mmlu_pro_law", "mmlu", "mmlu_pro/law_converted.jsonl"),
        "mmlu_pro_math": DatasetConfig("mmlu_pro_math", "mmlu", "mmlu_pro/math_converted.jsonl"),
        "mmlu_pro_other": DatasetConfig("mmlu_pro_other", "mmlu", "mmlu_pro/other_converted.jsonl"),
        "mmlu_pro_philosophy": DatasetConfig("mmlu_pro_philosophy", "mmlu", "mmlu_pro/philosophy_converted.jsonl"),
        "mmlu_pro_physics": DatasetConfig("mmlu_pro_physics", "mmlu", "mmlu_pro/physics_converted.jsonl"),
        "mmlu_pro_psychology": DatasetConfig("mmlu_pro_psychology", "mmlu", "mmlu_pro/psychology_converted.jsonl"),

        
        "math": DatasetConfig("math", "math", "math.jsonl"),
        "gsm8k": DatasetConfig("gsm8k", "math", "gsm8k.jsonl"),
        "aime24": DatasetConfig("aime24", "math", "AIME_2024.jsonl"),
        "mt_bench": DatasetConfig("mt_bench", "general", "mt_bench.jsonl"),
        "magpie": DatasetConfig("magpie", "general", "magpie.jsonl"),
        "numina_cot_5k": DatasetConfig("numina_cot_5k", "math", "numina_cot_5k.jsonl"),
        "mmlu_full": DatasetConfig("mmlu_full", "mmlu", "mmlu.jsonl"),
        "magpie_5k": DatasetConfig("magpie_5k", "general", "magpie_5k.jsonl"),
        "alpaca_5k": DatasetConfig("alpaca_5k","general","alpaca_5k.jsonl"),
        "dapo-math-17k_dedup": DatasetConfig("dapo-math-17k_dedup","math","dapo-math-17k_dedup.jsonl"),
        # for train
        # "mmlu_5k": DatasetConfig("mmlu_5k", "mmlu", "mmlu_5k.jsonl"),
        "mmlu_train": DatasetConfig("mmlu_train", "mmlu", "mmlu_train.jsonl"),
        "mmlu_test": DatasetConfig("mmlu_test", "mmlu", "mmlu_test.jsonl"),
        "numina_cot_5k_train": DatasetConfig("numina_cot_5k_train", "math", "numina_cot_5k_train.jsonl"),
        "numina_cot_5k_test": DatasetConfig("numina_cot_5k_test", "math", "numina_cot_5k_test.jsonl"),
        "magpie_5k_train": DatasetConfig("magpie_5k_train", "general", "magpie_5k_train.jsonl"),
        "magpie_5k_test": DatasetConfig("magpie_5k_test", "general", "magpie_5k_test.jsonl"),
        "alpaca_5k_train": DatasetConfig("alpaca_5k_train","general","alpaca_5k_train.jsonl"),
         "alpaca_5k_test": DatasetConfig("alpaca_5k_test","general","alpaca_5k_test.jsonl"),
         "metamath_5k_test": DatasetConfig("metamath_5k_test", "math", "metamath_5k_test.jsonl"),
         "metamath_5k_train": DatasetConfig("metamath_5k_train", "math", "metamath_5k_train.jsonl"),
         "big_math_5k_train": DatasetConfig("big_math_5k_train", "math", "big_math_5k_train.jsonl"),
         "big_math_5k_test": DatasetConfig("big_math_5k_test", "math", "big_math_5k_test.jsonl"),
    }

    @classmethod
    def register_dataset(cls, name: str, dataset_type: str, file_path: str):
        cls.DATASETS[name] = DatasetConfig(name, dataset_type, file_path)

    @classmethod
    def get_dataset(cls, name: str) -> DatasetConfig:
        if name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        return cls.DATASETS[name]

    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls.DATASETS.keys())


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.loss_calc = MultiModalLossCalculator(model=None, tokenizer=None, device=None)

    def load_jsonl(self, filepath: str) -> List[Dict]:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return data

    def load_dataset(self, dataset_name: str) -> Tuple[List[Dict], str]:
        config = DatasetRegistry.get_dataset(dataset_name)
        file_path = self.data_dir / config.file_path


        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        data = self.load_jsonl(file_path)
        return data, config.type

    def format_prompts(self, data: List[Dict], dataset_type: str) -> List[str]:
        return [self.loss_calc.format_prompt(item["instruction"], dataset_type) for item in data]


class ModelEvaluator:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.loss_calc = MultiModalLossCalculator(model=None, tokenizer=None, device=None)

    def evaluate_model(self, model_path: str, dataset_name: str,
                      max_tokens: int = 512, temperature: float = 0.0, 
                      model_type: str = "weak") -> List[Dict]:
        data, dataset_type = self.data_loader.load_dataset(dataset_name)
        prompts = self.data_loader.format_prompts(data, dataset_type)
   
        print(f"Generated {len(prompts)} prompts")
        responses = parallel_inference(
                prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                model_name_or_path=model_path,
                type=model_type
            )

        results = []
        
       
        if dataset_type == "general":
            conversations = []
            for item, response in zip(data, responses):
                conversation = [
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": response}
                ]
                conversations.append(conversation)
            
            reward_scores = evaluate_skywork_reward(conversations, device="cuda:0")

            scores = reward_scores
            
        results = []
        for i, (item, response) in enumerate(zip(data, responses)):
            instruction = item["instruction"]
            gold_response = item.get("response", "")
            
            if dataset_type == "general":
                score = scores[i]  
            else:
                is_correct = self.loss_calc._evaluate_response(response, gold_response, instruction, dataset_type)
                score = 1.0 if is_correct else 0.0
            
            
            result = {
                "id": i,
                "instruction": instruction,
                "response": gold_response,
                "generated_response": response,
                "score": score,
                "dataset": dataset_name,
                "dataset_type": dataset_type
            }
            results.append(result)
        return results

    def evaluate_pass_at_10_from_runs(self, 
                                run_files_dir: str,
                                dataset_name: str,
                                k: int = 10,
                                run_prefix: str = "run",
                                response_field: str = "large_response") -> List[Dict]:
        """
        Calculate pass@k: at least 1 correct in k attempts
        Returns: List of results, one per question
        """
        import json
        from pathlib import Path
        from collections import defaultdict
        
        run_dir = Path(run_files_dir)
        data, dataset_type = self.data_loader.load_dataset(dataset_name)
        
        # Read all run files
        all_responses_by_index = defaultdict(list)
        
        for run_id in range(100):
            run_file = run_dir / f"{dataset_name}_{run_prefix}{run_id}.jsonl"
            if not run_file.exists():
                break
            
            with open(run_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    index = item.get('index', item.get('id', -1))
                    if index == -1:
                        continue
                    response = item.get(response_field, '')
                    all_responses_by_index[index].append(response)
        
        # Calculate pass@k for each question
        results = []
        
        for i, item in enumerate(data):
            instruction = item["instruction"]
            gold_response = item.get("response", "")
            responses = all_responses_by_index.get(i, [])
            
            # Skip if not enough responses
            if len(responses) < k:
                results.append({
                    "id": i,
                    "instruction": instruction,
                    "pass_at_10": 0.0,  # 不够k次就算失败
                    "n_responses": len(responses)
                })
                continue
            
            # Check if at least 1 correct in first k responses
            has_correct = False
            for response in responses[:k]:
                if self.loss_calc._evaluate_response(response, gold_response, instruction, dataset_type):
                    has_correct = True
                    break
            
            results.append({
                "id": i,
                "instruction": instruction,
                "pass_at_10": 1.0 if has_correct else 0.0,
                "n_responses": k
            })
        
        return results

    def evaluate_model_from_file(self, small_file_path: str, large_file_path: str, dataset_name: str) -> Dict:
        """
        从文件中读取responses并进行evaluation
        """
        import json
        
        # 读取文件
        small_data = []
        large_data = []
        
        with open(small_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    small_data.append(json.loads(line))
        
        with open(large_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    large_data.append(json.loads(line))
        
        # 加载数据集
        data, dataset_type = self.data_loader.load_dataset(dataset_name)
        small_scores = []
        if dataset_type == "general":
            # Batch all general dataset evaluations for parallel processing
            questions = [{"instruction": item['instruction']} for item in data]
            answers = [{"response": entry.get('generated_response', '')} for entry in small_data]
            ref_answers = [{"response": item.get('response', '')} for item in data]
            small_scores = llm_judge_general(questions, answers, "gpt-5", ref_answers, max_workers=32, batch_size=8)
        else:
            # Process non-general datasets individually
            for item, entry in zip(data, small_data):
                response = entry.get('generated_response', '')
                gold_response = item.get('response', '')
                is_correct = self.loss_calc._evaluate_response(
                    response, gold_response, item['instruction'], dataset_type
                )
                score = 1.0 if is_correct else 0.0
                small_scores.append(score)
        
        # 评估大模型
        large_scores = []
        if dataset_type == "general":
            # Batch all general dataset evaluations for parallel processing
            questions = [{"instruction": item['instruction']} for item in data]
            answers = [{"response": entry.get('generated_response', entry.get('large_response', ''))} for entry in large_data]
            ref_answers = [{"response": item.get('response', '')} for item in data]
            large_scores = llm_judge_general(questions, answers, "gpt-5", ref_answers, max_workers=32, batch_size=8)
        else:
            # Process non-general datasets individually
            for item, entry in zip(data, large_data):
                response = entry.get('generated_response', entry.get('large_response', ''))
                gold_response = item.get('response', '')
                is_correct = self.loss_calc._evaluate_response(
                    response, gold_response, item['instruction'], dataset_type
                )
                score = 1.0 if is_correct else 0.0
                large_scores.append(score)
        
        # 构建结果
        small_results = []
        large_results = []
        
        for i, (item, s_entry, l_entry, s_score, l_score) in enumerate(
            zip(data, small_data, large_data, small_scores, large_scores)
        ):
            small_results.append({
                "id": i,
                "instruction": item["instruction"],
                "response": item.get("response", ""),
                "generated_response": s_entry.get('generated_response', ''),
                "score": s_score,
                "dataset": dataset_name,
                "dataset_type": dataset_type
            })
           
            
            large_results.append({
                "id": i,
                "instruction": item["instruction"],
                "response": item.get("response", ""),
                "generated_response": l_entry.get('generated_response', l_entry.get('large_response', '')),
                "score": l_score,
                "dataset": dataset_name,
                "dataset_type": dataset_type
            })
        
        small_accuracy = sum(small_scores) / len(small_scores)
        large_accuracy = sum(large_scores) / len(large_scores)
        with open(small_file_path, 'w', encoding='utf-8') as f:
            for result in small_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        with open(large_file_path,'w',encoding='utf-8') as f:
            for result in large_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        
        print(f"Small model accuracy: {small_accuracy:.3f}")
        print(f"Large model accuracy: {large_accuracy:.3f}")
        
        return {
            dataset_name: {
                "small_results": small_results,
                "large_results": large_results,
                "small_accuracy": small_accuracy,
                "large_accuracy": large_accuracy
            }
        }
     
    def evaluate_dataset(self, 
                     small_model_path: str, 
                     large_model_path: Optional[str],
                     dataset_name: str,
                     small_output_path: str,
                     large_output_path: str,
                     openai_api_base: Optional[str] = None,
                     openai_api_key: Optional[str] = None,
                     **kwargs) -> Dict:
        """
        评估数据集,支持只运行小模型或同时运行大小模型
        
        Args:
            small_model_path: 小模型路径
            large_model_path: 大模型路径,如果为None则只运行小模型
            dataset_name: 数据集名称
            small_output_path: 小模型输出路径
            large_output_path: 大模型输出路径
            openai_api_base: OpenAI API基础URL
            openai_api_key: OpenAI API密钥
            **kwargs: 其他参数
        
        Returns:
            与 evaluate_model_from_file 相同格式的字典
        """
        import json
        
        print(f"Evaluating {dataset_name}...")
        
        # 加载数据集
        data, dataset_type = self.data_loader.load_dataset(dataset_name)
        
        # 评估小模型(生成responses)
        small_generated = self.evaluate_model(
            small_model_path, 
            dataset_name, 
            model_type="weak",
            **kwargs
        )
        
        # 评估小模型 - 与 evaluate_model_from_file 相同的逻辑
        small_scores = []
        small_results = []
        
        # Batch evaluation for general datasets to improve performance
        if dataset_type == "general":
            # Prepare batch data for parallel evaluation
            questions = [{"instruction": item['instruction']} for item in data]
            answers = [{"response": entry.get('generated_response', '')} for entry in small_generated]
            ref_answers = [{"response": item.get('response', '')} for item in data]
            small_scores = llm_judge_general(questions, answers, "gpt-5", ref_answers, max_workers=32, batch_size=8)
        else:
            # Process non-general datasets individually
            for item, entry in zip(data, small_generated):
                response = entry.get('generated_response', '')
                gold_response = item.get('response', '')
                is_correct = self.loss_calc._evaluate_response(
                    response, gold_response, item['instruction'], dataset_type
                )
                score = 1.0 if is_correct else 0.0
                small_scores.append(score)
        
        # Create results for small model
        for i, (item, entry, score) in enumerate(zip(data, small_generated, small_scores)):
            small_results.append({
                "id": i,
                "instruction": item["instruction"],
                "response": item.get('response', ''),
                "generated_response": entry.get('generated_response', ''),
                "score": score,
                "dataset": dataset_name,
                "dataset_type": dataset_type
            })
        
        small_accuracy = sum(small_scores) / len(small_scores)
        
        # 保存小模型结果
        with open(small_output_path, 'w', encoding='utf-8') as f:
            for result in small_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # 评估大模型
        if large_model_path is not None:
            large_generated = self.evaluate_model(
                large_model_path, 
                dataset_name, 
                model_type="strong",
                openai_api_base=openai_api_base,
                openai_api_key=openai_api_key, 
                **kwargs
            )
            
            # 评估大模型 - 与 evaluate_model_from_file 相同的逻辑
            large_scores = []
            large_results = []
            
            # Batch evaluation for large model
            if dataset_type == "general":
                # Prepare batch data for parallel evaluation
                questions = [{"instruction": item['instruction']} for item in data]
                answers = [{"response": entry.get('generated_response', entry.get('large_response', ''))} for entry in large_generated]
                ref_answers = [{"response": item.get('response', '')} for item in data]
                large_scores = llm_judge_general(questions, answers, "gpt-5", ref_answers, max_workers=32, batch_size=8)
            else:
                # Process non-general datasets individually
                for item, entry in zip(data, large_generated):
                    response = entry.get('generated_response', entry.get('large_response', ''))
                    gold_response = item.get('response', '')
                    is_correct = self.loss_calc._evaluate_response(
                        response, gold_response, item['instruction'], dataset_type
                    )
                    score = 1.0 if is_correct else 0.0
                    large_scores.append(score)
            
            # Create results for large model
            for i, (item, entry, score) in enumerate(zip(data, large_generated, large_scores)):
                large_results.append({
                    "id": i,
                    "instruction": item["instruction"],
                    "response": item.get('response', ''),
                    "generated_response": entry.get('generated_response', entry.get('large_response', '')),
                    "score": score,
                    "dataset": dataset_name,
                    "dataset_type": dataset_type
                })
            
            large_accuracy = sum(large_scores) / len(large_scores)
            
            # 保存大模型结果
            with open(large_output_path, 'w', encoding='utf-8') as f:
                for result in large_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"{dataset_name} - Small model: {small_accuracy:.3f}, Large model: {large_accuracy:.3f}")
        else:
            # 没有大模型时返回空结果
            large_results = []
            large_accuracy = 0.0
            print(f"{dataset_name} - Small model: {small_accuracy:.3f}")
        
        # 返回与 evaluate_model_from_file 相同格式的结果
        return {
            dataset_name: {
                "small_results": small_results,
                "large_results": large_results,
                "small_accuracy": small_accuracy,
                "large_accuracy": large_accuracy
            }
        }

class DataManager:
    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        self.data_loader = DataLoader(data_dir)
        self.evaluator = ModelEvaluator(self.data_loader)
        self.output_dir = Path(output_dir)

    def evaluate_models_on_datasets(self, 
                                    small_model_path: str, 
                                    large_model_path: str,
                                  datasets: List[str], **kwargs) -> Dict[str, Dict]:
        results = {}

        small_model_name = Path(small_model_path).name
        large_model_name = Path(large_model_path).name

        small_dir = self.output_dir / small_model_name
        large_dir = self.output_dir / large_model_name
        small_dir.mkdir(parents=True, exist_ok=True)
        large_dir.mkdir(parents=True, exist_ok=True)

        for dataset in datasets:
            small_output = small_dir / f"{dataset}.jsonl"
            large_output = large_dir / f"{dataset}.jsonl"
            small_results, large_results = self.evaluator.evaluate_dataset(
                small_model_path=small_model_path, 
                large_model_path=large_model_path, 
                dataset_name=dataset,
                small_output_path=str(small_output),
                large_output_path=str(large_output),
                **kwargs
            )

            

            results[dataset] = {
                "small_results": small_results,
                "large_results": large_results,
                "small_accuracy": sum(r["score"] for r in small_results) / len(small_results),
                "large_accuracy": sum(r["score"] for r in large_results) / len(large_results)
            }

        return results


    def evaluate_single_model_on_dataset(self, 
                                    small_model_path: str, 
                                  datasets: List[str], **kwargs) -> Dict[str, Dict]:
        results = {}

        small_model_name = Path(small_model_path).name


        for dataset in datasets:
            small_results, large_results = self.evaluator.evaluate_dataset(
                small_model_path =small_model_path, dataset_name= dataset, **kwargs
            )

            # Save results
            small_dir = self.output_dir / small_model_name
            small_dir.mkdir(parents=True, exist_ok=True)

            self.evaluator.save_results(small_results, small_dir / f"{dataset}.jsonl")
         
            results[dataset] = {
                "small_results": small_results,
                "small_accuracy": sum(r["score"] for r in small_results) / len(small_results),
              
            }

        return results


    def load_model_results(self, model_name: str, dataset: str) -> List[Dict]:
        file_path = self.output_dir / model_name / f"{dataset}.jsonl"
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        return self.data_loader.load_jsonl(file_path)


def register_custom_dataset(name: str, dataset_type: str, file_path: str):
    """Register a new dataset for evaluation"""
    DatasetRegistry.register_dataset(name, dataset_type, file_path)


def list_available_datasets() -> List[str]:
    """List all available datasets"""
    return DatasetRegistry.list_datasets()
