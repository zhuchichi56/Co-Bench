try:
    import fire  # type: ignore
except ModuleNotFoundError:
    fire = None
import numpy as np
from typing import Dict, Optional
from data import DataManager, DatasetRegistry, register_custom_dataset, list_available_datasets
from router import RouterManager, get_available_probe_types
from metric import BatchMetricEvaluator
from config import PipelineConfig
import json
import os
from loss_calculator import llm_as_a_judge


class RouterEvaluationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_manager = DataManager(config.data_dir, config.output_dir, inference_config=config.inference)
        self.router_manager = RouterManager()
        self.metric_evaluator = BatchMetricEvaluator(config.metric_results_dir)

    def _load_torch_data(self, file_path: str):
        """
        Load data from a torch file and extract data if it's wrapped in a dict.
        
        Args:
            file_path: Path to the torch file
            
        Returns:
            The loaded data (extracted from dict if necessary)
        """
        import torch
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return data

    def _filter_data_by_indices(self, data, valid_indices):
        """
        Filter data by valid indices.
        
        Args:
            data: List or array of data to filter
            valid_indices: Indices to keep (can be range, list, or iterable)
            
        Returns:
            Filtered data list
        """
        if hasattr(valid_indices, "__iter__") and not isinstance(valid_indices, str):
            return [data[i] for i in valid_indices]
        return data

    def _read_jsonl_file(self, file_path: str):
        """
        Read JSONL file and return list of parsed JSON objects.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of parsed JSON objects
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _setup_simple_router(self, router_type: str, router_config: Dict, datasets: list, 
                             model_results: Dict, router_scores_dict: Dict, router_identifier: str = None):
        """
        Setup router for simple router types that only need model_results.
        Handles: self_questioning, deberta, trained_deberta, llm, logits_margin, semantic_entropy.
        
        Args:
            router_type: Type of router to create
            router_config: Router configuration dictionary
            datasets: List of dataset names
            model_results: Dictionary containing model evaluation results
            router_scores_dict: Dictionary to store router scores (will be modified)
            router_identifier: Optional router identifier (defaults to router_type)
            
        Returns:
            Router identifier string
        """
        router_creation_map = {
            "self_questioning": lambda: self.router_manager.create_self_questioning_router(
                router_config["model_path"]
            ),
            "deberta": lambda: self.router_manager.create_deberta_router(
                router_config["model_path"]
            ),
            "trained_deberta": lambda: self.router_manager.create_trained_deberta_router(
                router_config["model_path"]
            ),
            "llm": lambda: self.router_manager.create_llm_router(
                router_config["model_path"]
            ),
            "logits_margin": lambda: self.router_manager.create_logits_margin_router(
                router_config["model_path"]
            ),
            "semantic_entropy": lambda: self.router_manager.create_semantic_entropy_router(
                router_config["model_path"],
                num_samples=router_config.get("num_samples", 5)
            ),
        }

        if router_type not in router_creation_map:
            raise ValueError(f"Unsupported router type for _setup_simple_router: {router_type}")

        router_name = router_creation_map[router_type]()
        router_identifier = router_identifier or router_type

        for dataset in datasets:
            small_results = model_results[dataset]["small_results"]
            router_scores = self.router_manager.get_router_scores(router_name, small_results, model_type="weak")
            router_scores_dict[dataset] = router_scores

        return router_identifier
        
    def get_score(self, task: str, judge_model: str = "gpt-5",
                  question_file: str = None, ref_answer_file: str = None) -> str:
        """
        Get scores for a specific task by evaluating models and comparing their performance
        Returns the path to the saved {task}.jsonl file

        Args:
            task: Task name (e.g., 'aime24', 'mt-bench')
            judge_model: Judge model for MT-Bench (default: gpt-5)
            question_file: Custom question file for MT-Bench
            ref_answer_file: Reference answer file for MT-Bench
        """
        print(f"Getting scores for task: {task}")

        # Check if this is MT-Bench
        is_mt_bench = task.lower() in ['mt-bench', 'mtbench', 'mt_bench']

        if is_mt_bench:
            return self._get_mt_bench_score(judge_model, question_file, ref_answer_file)

        # Original logic for non-MT-Bench tasks
        datasets = [task]
        small_model_path = self.config.inference.weak_model_path
        large_model_path = self.config.inference.strong_model_path
        max_tokens = self.config.inference.max_tokens
        temperature = self.config.inference.temperature
        output_path = self.config.output_dir
        small_model_name = os.path.basename(small_model_path)
        large_model_name = os.path.basename(large_model_path)
        small_model_output_dir = os.path.join(output_path, small_model_name)
        large_model_output_dir = os.path.join(output_path, large_model_name)
        if task.startswith("mmlu_pro"):
            small_model_output_dir = os.path.join(output_path,"mmlu_pro", small_model_name)
            large_model_output_dir = os.path.join(output_path,"mmlu_pro", large_model_name)
        small_output_file = os.path.join(small_model_output_dir, f"{task}.jsonl")
        large_output_file = os.path.join(large_model_output_dir, f"{task}.jsonl")
        print(f"Small model output: {small_output_file}")
        print(f"Large model output: {large_output_file}")

        # Create output directory
        os.makedirs(small_model_output_dir, exist_ok=True)
        os.makedirs(large_model_output_dir, exist_ok=True)

        # Evaluate small model
        if os.path.exists(small_output_file):
            print(f"Found existing small model results, loading from file...")
            small_result = self.data_manager.evaluator.evaluate_single_model_from_file(
                small_output_file,
                task,
                model_type="weak"
            )
        else:
            print(f"Evaluating small model...")
            small_result = self.data_manager.evaluator.evaluate_single_dataset(
                model_path=small_model_path,
                dataset_name=task,
                output_path=small_output_file,
                model_type="weak",
                max_tokens=max_tokens,
                temperature=temperature
            )

        small_results = small_result["results"]
        small_accuracy = small_result["accuracy"]

        # Evaluate large model
        if os.path.exists(large_output_file):
            print(f"Found existing large model results, loading from file...")
            large_result = self.data_manager.evaluator.evaluate_single_model_from_file(
                large_output_file,
                task,
                model_type="strong"
            )
        else:
            print(f"Evaluating large model...")
            large_result = self.data_manager.evaluator.evaluate_single_dataset(
                model_path=large_model_path,
                dataset_name=task,
                output_path=large_output_file,
                model_type="strong",
                max_tokens=max_tokens,
                temperature=temperature
            )

        large_results = large_result["results"]
        large_accuracy = large_result["accuracy"]

        print(f"\nResults: Small model: {small_accuracy:.3f}, Large model: {large_accuracy:.3f}")
        
    
        output_file = os.path.join(output_path, f"{task}.jsonl")
        output_data = []

        for  small_result,large_result in zip(small_results,large_results):
            small_score = small_result.get("score", 0.0)
            large_score = large_result.get("score", 0.0)
            if task.startswith('alpaca') or task.startswith('magpie'): #label               
                score = 1 if small_score >= large_score else 0
            else:
               
                score = small_score

            entry = {
                "instruction": small_result.get("instruction", ""),
                "small_response": small_result.get("generated_response", ""),
                "large_response": large_result.get("generated_response", ""),
                "score": float(score)
            }
            output_data.append(entry)

          
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Saved {len(output_data)} samples to {output_file}")
        return output_file

    def _get_mt_bench_score(self, judge_model: str = "gpt-5",
                        question_file: str = None, ref_answer_file: str = None) -> str:
        """Internal method for MT-Bench evaluation"""
        if question_file is None:
            question_file = "data/llmjudge/mt_bench/question.jsonl"

        # Set up paths
        small_model_path = self.config.inference.weak_model_path
        model_name = os.path.basename(small_model_path)
        output_path = self.config.output_dir
        model_output_dir = os.path.join(output_path, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        output_file = os.path.join(model_output_dir, "mt-bench.jsonl")
        
        # Check for existing response file (saved by generate_multi_turn_responses)
        responses_file = os.path.join("./results/gpt-5", "mt-bench.jsonl")
        
        if os.path.exists(responses_file):
            print(f"Found existing MT-Bench responses at {responses_file}")
            print("Skipping generation, proceeding to evaluation...")
            
            # Load questions and existing responses
            questions = self._read_jsonl_file(question_file)
            responses_data = self._read_jsonl_file(responses_file)
            
            # Format for judge evaluation
            small_answers = []
            for response_data in responses_data:
                small_answers.append({
                    "question_id": response_data.get("question_id"),
                    "choices": response_data.get("choices", [])
                })
            
            # Run evaluation with LLM-as-a-Judge
            print(f"Evaluating with {judge_model}...")
            small_scores = llm_as_a_judge(questions, small_answers, judge_model, ref_answer_file)
            
        else:
            # Run MT-Bench evaluation (generation + evaluation)
            print("No existing responses found, running full evaluation...")
            results = self.evaluate_mt_bench(question_file, judge_model, ref_answer_file)
            small_scores = results.get("small_scores", [])
            
            # Load questions for formatting
            questions = self._read_jsonl_file(question_file)

        # Convert to standard format and save final results
        output_data = []
        for i, (question, score) in enumerate(zip(questions, small_scores)):
            # Normalize score to [0, 1]
            normalized_score = score / 10.0 if score > 0 else 0.0

            entry = {
                "instruction": question["turns"][0],  # Use first turn as instruction
                "small_response": "",  # Response is stored separately in MT-Bench format
                "score": float(normalized_score)
            }
            output_data.append(entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in output_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Saved {len(output_data)} MT-Bench evaluation results to {output_file}")
        return output_file

    def evaluate_complete_pipeline(self, hidden_states_file: Optional[str], datasets, query_embeddings_file: Optional[str] = None) -> Dict:
        """
        Simplified evaluation using only config parameters
        """
        # Use all config parameters
        small_model_path = self.config.inference.weak_model_path
        large_model_path = self.config.inference.strong_model_path
        small_model_name = os.path.basename(small_model_path)
        if large_model_path == "false":
            large_model_name = "gpt-5"
        else:
            large_model_name = os.path.basename(large_model_path)
        
        recovery_rate_band = self.config.recovery_rate_band
        lpm_call_rate_band = self.config.lpm_call_rate_band
        router_config = self.config.router.to_dict(self.config.inference)

        print("Starting pipeline evaluation")
        print(f"Datasets: {datasets}")
        print(f"Router Type: {router_config.get('type', 'unknown')}")

        # Step 1: Evaluate models on datasets
        dataset_name = datasets[0]
        if dataset_name.startswith("mmlu_pro"):
            results_dir = "results/mmlu_pro"
        else:
            results_dir = "results"

        small_file = os.path.join(results_dir, f"{small_model_name}", f"{dataset_name}.jsonl")
        large_file = os.path.join(results_dir, f"{large_model_name}", f"{dataset_name}.jsonl")
        print(f"small_file: {small_file}")

        if os.path.exists(small_file) and os.path.exists(large_file):
            print(f"Found existing results, loading...")
            
            small_results = self._read_jsonl_file(small_file)
            large_results = self._read_jsonl_file(large_file)

        # Compute accuracy (use full data)
        small_accuracy = sum(r["score"] for r in small_results) / len(small_results) if small_results else 0.0
        large_accuracy = sum(r["score"] for r in large_results) / len(large_results) if large_results else 0.0

        print(f"{datasets[0]} - Small model: {small_accuracy:.3f}, Large model: {large_accuracy:.3f}")

        model_results = {}
        dataset_type = "general"
        try:
            dataset_type = DatasetRegistry.get_dataset(datasets[0]).type
        except Exception:
            dataset_type = "general"

        model_results[datasets[0]] = {
            "small_results": small_results,
            "large_results": large_results,
            "small_accuracy": small_accuracy,
            "large_accuracy": large_accuracy,
            "dataset_type": dataset_type,
            "valid_indices": list(range(len(small_results)))  # all samples are valid
        }

        # Step 2: Generate router scores
        print(f"Step 2: Generating router scores using {router_config['type']} router")
        router_scores_dict = {}
        router_identifier = None
        
        # Setup router based on config
        if router_config["type"] == "probe":
            if hidden_states_file is None:
                raise ValueError("hidden_states_file required for probe router")

            probe_type = router_config.get("probe_type")

            # All probe types (including dynamic_softmax and dynamic_dirichlet) are handled by ProbeRouter
            router_name = self.router_manager.create_probe_router(
                router_config["checkpoint_path"],
                probe_type
            )
            router_identifier = "dynamic" if probe_type in ["dynamic_softmax", "dynamic_dirichlet"] else router_config["type"]

            hidden_data = self._load_torch_data(hidden_states_file)

            for dataset in datasets:
                valid_indices = model_results[dataset].get("valid_indices", range(len(hidden_data)))
                filtered_hidden_data = self._filter_data_by_indices(hidden_data, valid_indices)
                
                router_scores = self.router_manager.get_router_scores(router_name, filtered_hidden_data)
                router_scores_dict[dataset] = router_scores

        elif router_config["type"] in ["self_questioning", "deberta", "trained_deberta", "llm", "logits_margin", "semantic_entropy"]:
            router_identifier = self._setup_simple_router(
                router_config["type"],
                router_config,
                datasets,
                model_results,
                router_scores_dict
            )

        elif router_config["type"] == "embedding_mlp":
            router_name = self.router_manager.create_embedding_mlp_router(
                router_config["checkpoint_path"]
            )
            router_identifier = "embedding_mlp"

            if query_embeddings_file is None:
                raise ValueError("query_embeddings_file is required for embedding_mlp router")

            embedding_data = self._load_torch_data(query_embeddings_file)
            if not embedding_data:
                raise ValueError(f"Embedding file {query_embeddings_file} is empty or missing 'data'")

            for dataset in datasets:
                valid_indices = model_results[dataset].get("valid_indices", range(len(embedding_data)))
                # Validate indices
                if hasattr(valid_indices, "__iter__") and not isinstance(valid_indices, str):
                    max_idx = max(valid_indices)
                    if max_idx >= len(embedding_data):
                        raise IndexError(
                            f"Embedding index out of range for dataset {dataset}: "
                            f"max_idx={max_idx}, embedding_len={len(embedding_data)}, file={query_embeddings_file}"
                        )
                filtered_embeddings = self._filter_data_by_indices(embedding_data, valid_indices)
                router_scores = self.router_manager.get_router_scores(router_name, filtered_embeddings)
                router_scores_dict[dataset] = router_scores


        elif router_config["type"] == "coe":
            router_name = self.router_manager.create_coe_router()
            router_identifier = "coe"

            if hidden_states_file is None:
                raise ValueError("hidden_states_file required for coe router")

            hidden_data = self._load_torch_data(hidden_states_file)

            for dataset in datasets:
                valid_indices = model_results[dataset].get("valid_indices", range(len(hidden_data)))
                filtered_hidden_data = self._filter_data_by_indices(hidden_data, valid_indices)
                
                router_scores = self.router_manager.get_router_scores(router_name, filtered_hidden_data)
                router_scores_dict[dataset] = router_scores

       
        elif router_config["type"] in ["max_logits", "top10_variance", "entropy", "confidence_margin"]:
            if router_config["type"] == "max_logits":
                router_name = self.router_manager.create_max_logits_router()
            elif router_config["type"] == "top10_variance":
                router_name = self.router_manager.create_top10_variance_router()
            elif router_config["type"] == "entropy":
                router_name = self.router_manager.create_entropy_router()
            elif router_config["type"] == "confidence_margin":
                router_name = self.router_manager.create_confidence_margin_router()

            router_identifier = router_config["type"]

            for dataset in datasets:
                from pathlib import Path
                logits_output_dir = Path(self.config.training.logits_output_dir or "logits_output")
                weak_model_name = os.path.basename(self.config.inference.weak_model_path)

                if dataset.startswith("mmlu_pro_"):
                    logits_file = logits_output_dir / "mmlu_pro" / f"{weak_model_name}_{dataset}.pt"
                else:
                    logits_file = logits_output_dir / f"{weak_model_name}_{dataset}.pt"

                if not logits_file.exists():
                    print(f"⚠️ Warning: Logits file {logits_file} not found, using default scores")
                    small_results = model_results[dataset]["small_results"]
                    router_scores = np.array([item.get("score", 0.5) for item in small_results])
                    router_scores_dict[dataset] = router_scores
                    continue

                logits_data = self._load_torch_data(str(logits_file))

                valid_indices = model_results[dataset].get("valid_indices", range(len(logits_data)))
                filtered_logits_data = self._filter_data_by_indices(logits_data, valid_indices)

                router_scores = self.router_manager.get_router_scores(router_name, filtered_logits_data)
                router_scores_dict[dataset] = router_scores

        # Step 3: Calculate metrics
        print("Step 3: Calculating metrics")
        metric_results = self.metric_evaluator.evaluate_multiple_datasets(
            model_results, router_scores_dict,
            router=router_identifier,
            recovery_rate_band=recovery_rate_band,
            lpm_call_rate_band=lpm_call_rate_band
        )

        print("Pipeline evaluation complete!")

        for dataset, scores in router_scores_dict.items():
            avg_score = np.mean(scores)
            print(f"{dataset}: Average router score = {avg_score:.3f}")

        if metric_results:
            for dataset, metrics in metric_results.items():
                if isinstance(metrics, dict):
                    reliable_metrics = metrics.get('reliable_metrics', 'N/A')
                    auroc = reliable_metrics.get('auroc', 'N/A')
                    print(f"{dataset}: AUROC = {auroc}")

        return {
            "model_results": model_results,
            "router_scores": {k: v.tolist() for k, v in router_scores_dict.items()},
            "metric_results": metric_results
        }        

    def register_dataset(self, name: str, dataset_type: str, file_path: str):
        """Register a new dataset"""
        register_custom_dataset(name, dataset_type, file_path)
        print(f"Registered dataset: {name} (type: {dataset_type})")

    def list_datasets(self):
        """List available datasets"""
        return list_available_datasets()

    def list_probe_types(self):
        """List available probe types"""
        return get_available_probe_types()

    def evaluate_mt_bench(self, question_file: str, judge_model: str = "gpt-5",
                         ref_answer_file: str = None) -> Dict:
        """
        Evaluate MT-Bench using LLM-as-a-Judge

        Args:
            question_file: Path to MT-Bench questions JSONL file
            judge_model: Judge model name (default: gpt-5)
            ref_answer_file: Optional reference answers file

        Returns:
            Dict with evaluation results and scores
        """
        print(f"Starting MT-Bench evaluation with judge model: {judge_model}")

        # Load questions and reference answers if provided
        questions = self._read_jsonl_file(question_file)
        ref_answers = self._read_jsonl_file(ref_answer_file) if ref_answer_file else None

        # Generate answers using both models
        small_model_path = self.config.inference.weak_model_path

        large_model_path = self.config.inference.strong_model_path

        print(f"Evaluating {len(questions)} MT-Bench questions")

        # Use existing model evaluation infrastructure
        from inference.vllm_client import parallel_inference

        def generate_multi_turn_responses(model_path, questions):
            """Generate responses for multi-turn conversations using proper conversation templates"""
            all_responses = []

            # Temperature configuration (identical to original MT-Bench)
            TEMPERATURE_CONFIG = {
                "writing": 0.7,
                "roleplay": 0.7,
                "extraction": 0.0,
                "math": 0.0,
                "coding": 0.0,
                "reasoning": 0.0,
                "stem": 0.1,
                "humanities": 0.1,
                "arena-hard-200": 0.0,
            }

            for question in questions:
                turns = []
                category = question.get("category", "general")
                temperature = TEMPERATURE_CONFIG.get(category, 0.7)

                for turn_idx, turn_question in enumerate(question["turns"]):
                    # Use proper conversation template approach like original MT-Bench
                    if turn_idx == 0:
                        # First turn: format as single user question
                        # Use chat template format for better model compatibility
                        messages = [{"role": "user", "content": turn_question}]
                        prompt = self._format_chat_template(messages, model_path)
                    else:
                        # Multi-turn: build complete conversation history
                        messages = []
                        for i in range(turn_idx):
                            messages.append({"role": "user", "content": question["turns"][i]})
                            messages.append({"role": "assistant", "content": turns[i]})
                        messages.append({"role": "user", "content": turn_question})
                        prompt = self._format_chat_template(messages, model_path)

                    # Generate response for this turn
                    response = parallel_inference(
                        prompt_list=[prompt],
                        max_tokens=self.config.inference.max_tokens,
                        temperature=temperature,
                        model_name_or_path=model_path
                    )[0]

                    # Clean response (similar to original MT-Bench post-processing)
                    response = self._clean_response(response)
                    turns.append(response)

                all_responses.append(turns)
                
            output_dir = "./results/gpt-5/"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "mt-bench.jsonl")

            with open(output_file, 'w', encoding='utf-8') as f:
                for i, response_turns in enumerate(all_responses):
                    # Get corresponding question info
                    question = questions[i] if i < len(questions) else {}
                    
                    # Build output entries following the MT-Bench standard format
                    output_entry = {
                        "question_id": question.get("question_id", i + 1),
                        "model_id": os.path.basename(model_path),
                        "choices": [{"index": 0, "turns": response_turns}],
                    }
                    
                    f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

            return all_responses

        # Generate small model answers
        print("Generating small model responses...")
        small_responses = generate_multi_turn_responses(small_model_path, questions)

        # Generate large model answers
        print("Generating large model responses...")

        large_responses = generate_multi_turn_responses(large_model_path, questions)

        # Format answers for judge evaluation
        small_answers = []
        large_answers = []

        for i, question in enumerate(questions):
            small_answers.append({
                "question_id": question["question_id"],
                "choices": [{"turns": small_responses[i]}]
            })

            large_answers.append({
                "question_id": question["question_id"],
                "choices": [{"turns": large_responses[i]}]
            })

        # Evaluate with LLM-as-a-Judge
        print("Evaluating small model with LLM-as-a-Judge...")
        small_scores = llm_as_a_judge(questions, small_answers, judge_model, ref_answers)


        # print("Evaluating large model with LLM-as-a-Judge...")
        large_scores = llm_as_a_judge(questions, large_answers, judge_model, ref_answers)

        # Calculate statistics
        valid_small_scores = [s for s in small_scores if s > 0]
        valid_large_scores = [s for s in large_scores if s > 0]

        results = {
            "questions": len(questions),
            "small_model": small_model_path,
            "large_model": large_model_path,
            "judge_model": judge_model,
            "small_scores": small_scores,
            "large_scores": large_scores,
            "small_avg": sum(valid_small_scores) / len(valid_small_scores) if valid_small_scores else 0,
            "large_avg": sum(valid_large_scores) / len(valid_large_scores) if valid_large_scores else 0,
            "small_valid": len(valid_small_scores),
            "large_valid": len(valid_large_scores)
        }

        print(f"Small model average score: {results['small_avg']:.2f} ({results['small_valid']}/{results['questions']} valid)")
        print(f"Large model average score: {results['large_avg']:.2f} ({results['large_valid']}/{results['questions']} valid)")

        return results

    def _format_chat_template(self, messages, model_path):
        """Format messages using appropriate chat template"""
        # For GPT models, use direct message format
        if "gpt" in model_path.lower():
            return json.dumps(messages)

        # For other models, try to use a standard chat template format
        # This mimics the behavior of get_conversation_template() in original MT-Bench
        if "llama" in model_path.lower():
            # Llama-style formatting
            formatted = ""
            for msg in messages:
                if msg["role"] == "user":
                    formatted += f"<s>[INST] {msg['content']} [/INST]"
                elif msg["role"] == "assistant":
                    formatted += f"{msg['content']}</s>"
            return formatted
        elif "qwen" in model_path.lower():
            # Qwen-style formatting
            formatted = ""
            for msg in messages:
                if msg["role"] == "user":
                    formatted += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg["role"] == "assistant":
                    formatted += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            if not formatted.endswith("<|im_start|>assistant\n"):
                formatted += "<|im_start|>assistant\n"
            return formatted
        else:
            # Default format - try vllm_client's template handling
            conversation = []
            for msg in messages:
                if msg["role"] == "user":
                    conversation.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    conversation.append(f"Assistant: {msg['content']}")
            return "\n\n".join(conversation) + "\n\nAssistant:"

    def _clean_response(self, response):
            """Clean response similar to original MT-Bench"""
            if not response:
                return ""

            # Remove common prefixes that models might add
            response = response.strip()
            if response.startswith("Assistant:"):
                response = response.replace("Assistant:", "", 1).strip()

            # Remove special tokens (basic cleanup)
            special_tokens = ["<|im_end|>", "</s>", "<s>", "[/INST]"]
            for token in special_tokens:
                response = response.replace(token, "")

            return response.strip()

def evaluate_pipeline(config: PipelineConfig, hidden_states_file: str = None, datasets=None, query_embeddings_file: Optional[str] = None):
    """
    Main evaluation function using config parameters only
    """
    pipeline = RouterEvaluationPipeline(config)
    return pipeline.evaluate_complete_pipeline(hidden_states_file, datasets, query_embeddings_file=query_embeddings_file)


def get_task_score(config: PipelineConfig, task: str, judge_model: str = "gpt-5",
                   question_file: str = None, ref_answer_file: str = None):
    """
    Get scores for a specific task (unified interface)

    Args:
        config: Pipeline configuration
        task: Task name (e.g., 'aime24', 'mt-bench')
        judge_model: Judge model for MT-Bench (default: gpt-5)
        question_file: Custom question file for MT-Bench
        ref_answer_file: Reference answer file for MT-Bench

    Examples:
        # Regular task
        get_task_score(config, 'aime24')
    """
    pipeline = RouterEvaluationPipeline(config)
    return pipeline.get_score(task, judge_model, question_file, ref_answer_file)



if __name__ == "__main__":
    if fire is None:
        raise SystemExit("Missing optional dependency 'fire'. Install with: pip install fire")
    fire.Fire()
