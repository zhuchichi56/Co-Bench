import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loss_calculator import MultiModalLossCalculator
from inference.vllm_client import parallel_inference
from loss_calculator import llm_judge_general


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
        "big_math_10k": DatasetConfig("big_math_10k","math","big_math_10k.jsonl"),
        "alpaca_10k" :DatasetConfig("alpaca_10k","general","alpaca_10k.jsonl"),
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
         "test": DatasetConfig("test", "general", "test.jsonl"),
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
    def __init__(self, data_dir: str = "data", inference_config=None):
        self.data_dir = Path(data_dir)
        self.loss_calc = MultiModalLossCalculator(model=None, tokenizer=None, device=None, inference_config=inference_config)

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
    def __init__(self, data_loader: DataLoader, inference_config=None):
        self.data_loader = data_loader
        self.loss_calc = MultiModalLossCalculator(model=None, tokenizer=None, device=None, inference_config=inference_config)
        # Extract max_workers from inference_config if available
        self.max_workers = inference_config.max_workers if inference_config and hasattr(inference_config, 'max_workers') else 32 

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

        # # Evaluate responses
        # scores = []
        # if dataset_type == "general":
        #     # Use llm_judge_general for general datasets (consistent with evaluate_single_dataset)
        #     questions = [{"instruction": item['instruction']} for item in data]
        #     answers = [{"response": response} for response in responses]
        #     ref_answers = [{"response": item.get('response', '')} for item in data]
        #     scores = llm_judge_general(questions, answers, "gpt-5", ref_answers, max_workers=self.max_workers)
        # else:
        #     # Process non-general datasets individually
        #     for item, response in zip(data, responses):
        #         gold_response = item.get('response', '')
        #         is_correct = self.loss_calc._evaluate_response(
        #             response, gold_response, item['instruction'], dataset_type
        #         )
        #         score = 1.0 if is_correct else 0.0
        #         scores.append(score)

        # Build results
        results = []
        for i, (item, response) in enumerate(zip(data, responses)):
            result = {
                "id": i,
                "instruction": item["instruction"],
                "response": item.get("response", ""),
                "generated_response": response,
                # "score": score,
                "dataset": dataset_name,
                "dataset_type": dataset_type
            }
            results.append(result)
        return results

    def evaluate_single_model_from_file(self, file_path: str, dataset_name: str, model_type: str = "weak") -> Dict:
        """
        从文件中读取单个模型的responses并进行evaluation

        Args:
            file_path: 模型输出文件路径
            dataset_name: 数据集名称
            model_type: 模型类型 ("weak" 或 "strong")

        Returns:
            包含评估结果的字典: {"results": List[Dict], "accuracy": float}
        """
        import json

        # 读取文件
        model_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    model_data.append(json.loads(line))

        # 加载数据集
        data, dataset_type = self.data_loader.load_dataset(dataset_name)

        def _ensure_general_scores(entries: List[Dict], response_getter) -> List[float]:
            """Reuse existing scores when available and only rejudge missing ones."""
            scores = [entry.get('score', -1) for entry in entries]
            need_judge = [idx for idx, score in enumerate(scores) if score == -1]
            if not need_judge:
                return scores
            questions = [{"instruction": data[i]['instruction']} for i in need_judge]
            answers = [{"response": response_getter(entries[i])} for i in need_judge]
            ref_answers = [{"response": data[i].get('response', '')} for i in need_judge]
            judged_scores = llm_judge_general(questions, answers, "gpt-5", ref_answers, self.max_workers)
            for idx, judged in zip(need_judge, judged_scores):
                scores[idx] = judged
            return scores

        # 评估模型
        scores = []
        if dataset_type == "general":
            scores = _ensure_general_scores(
                model_data,
                lambda entry: entry.get('generated_response', entry.get('large_response', ''))
            )
        else:
            # Process non-general datasets individually
            for item, entry in zip(data, model_data):
                response = entry.get('generated_response', entry.get('large_response', ''))
                gold_response = item.get('response', '')
                is_correct = self.loss_calc._evaluate_response(
                    response, gold_response, item['instruction'], dataset_type
                )
                score = 1.0 if is_correct else 0.0
                scores.append(score)

        # 构建结果
        results = []
        for i, (item, entry, score) in enumerate(zip(data, model_data, scores)):
            results.append({
                "id": i,
                "instruction": item["instruction"],
                "response": item.get("response", ""),
                "generated_response": entry.get('generated_response', entry.get('large_response', '')),
                "score": score,
                "dataset": dataset_name,
                "dataset_type": dataset_type
            })

        accuracy = sum(scores) / len(scores) if scores else 0.0

        # 保存更新后的结果（带有score）
        with open(file_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"{model_type.capitalize()} model accuracy: {accuracy:.3f}")

        return {
            "results": results,
            "accuracy": accuracy
        }


    def evaluate_single_dataset(self,
                               model_path: str,
                               dataset_name: str,
                               output_path: str,
                               model_type: str = "weak",
                               **kwargs) -> Dict:
        """
        评估单个模型在数据集上的表现

        Args:
            model_path: 模型路径
            dataset_name: 数据集名称
            output_path: 输出文件路径
            model_type: 模型类型 ("weak" 或 "strong")
            **kwargs: 其他参数（如 max_tokens, temperature）

        Returns:
            {"results": List[Dict], "accuracy": float}
        """
        import json

        print(f"Evaluating {model_type} model on {dataset_name}...")

        # 加载数据集
        data, dataset_type = self.data_loader.load_dataset(dataset_name)

        # 生成responses
        generated = self.evaluate_model(
            model_path,
            dataset_name,
            model_type=model_type,
            **kwargs
        )

        # 对于 general 数据，推理完成后立即保存（防止后续联网失败）
        if dataset_type == "general":
            import os
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in generated:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"✓ [步骤1/2] 已保存推理结果到 {output_path}（共 {len(generated)} 条，暂不含最终 score）")

        # 评估模型
        scores = []
        results = []

        # Batch evaluation for general datasets
        if dataset_type == "general":
            questions = [{"instruction": item['instruction']} for item in data]
            answers = [{"response": entry.get('generated_response', '')} for entry in generated]
            ref_answers = [{"response": item.get('response', '')} for item in data]
            scores = llm_judge_general(questions, answers, "gpt-5", ref_answers, max_workers=self.max_workers)
        else:
            # Process non-general datasets individually
            for item, entry in zip(data, generated):
                response = entry.get('generated_response', '')
                gold_response = item.get('response', '')
                is_correct = self.loss_calc._evaluate_response(
                    response, gold_response, item['instruction'], dataset_type
                )
                score = 1.0 if is_correct else 0.0
                scores.append(score)

        # Create results
        for i, (item, entry, score) in enumerate(zip(data, generated, scores)):
            results.append({
                "id": i,
                "instruction": item["instruction"],
                "response": item.get('response', ''),
                "generated_response": entry.get('generated_response', ''),
                "score": score,
                "dataset": dataset_name,
                "dataset_type": dataset_type
            })

        accuracy = sum(scores) / len(scores) if scores else 0.0

        # 保存最终结果
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


        print(f"{dataset_name} - {model_type.capitalize()} model: {accuracy:.3f}")

        return {
            "results": results,
            "accuracy": accuracy
        }

class DataManager:
    def __init__(self, data_dir: str = "data", output_dir: str = "results", inference_config=None):
        self.data_loader = DataLoader(data_dir, inference_config=inference_config)
        self.evaluator = ModelEvaluator(self.data_loader, inference_config=inference_config)
        self.output_dir = Path(output_dir)
        # self.judge_model = 
        # self.max_workers = 
        
        

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
