from config import PipelineConfig
from pipeline import RouterEvaluationPipeline,get_task_score
from train_router import generate_logits,set_random_seed,complete_probe_training_pipeline_with_mixed_datasets,complete_layerwise_probe_training_pipeline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
import copy
import json
import argparse
import glob
import re


def batch_evaluate_probes(base_config, probe_configs, eval_tasks):
    """
    批量评估不同probe配置
    
    Args:
        base_config: 基础配置对象
        probe_configs: probe配置列表
        eval_tasks: 评估任务列表
    """
    all_results = {}
    
    for i, probe_config in enumerate(probe_configs):
       
        
        config_copy = copy.deepcopy(base_config)
        config_copy.router.checkpoint_path = probe_config['checkpoint_path']
        config_copy.router.probe_type = probe_config['probe_type']
        config_copy.metric_results_dir = probe_config['metric_results_dir']
        print(f"\n{'='*60}")
        print(f"Running evaluation {i+1}/{len(probe_configs)}")
        print(f"Probe type: {probe_config['probe_type']}")
        print(f"Checkpoint: {probe_config['checkpoint_path']}")
        print(f"{'='*60}")
        pipeline = RouterEvaluationPipeline(config_copy)
            
        for task in eval_tasks:
                print(f"\nEvaluating task: {task}")
                
                hidden_states_file = _build_hs_path(task)
                
                if not os.path.exists(hidden_states_file):
                    print(f"Warning: Hidden states file not found: {hidden_states_file}")
                    continue
                
                datasets = [f"{task}"]
                
                
                pipeline.evaluate_complete_pipeline(
                        hidden_states_file=hidden_states_file,
                        datasets=datasets
                    )
                
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="eval_batch_max_k")
    args = parser.parse_args()
    mode = args.mode
    set_random_seed(42)
    config = PipelineConfig().from_yaml()
    pipeline = RouterEvaluationPipeline(config)
    
    mmlu_pro_tasks = [
        "mmlu_pro_biology", "mmlu_pro_business",
        "mmlu_pro_chemistry", "mmlu_pro_computer_science",
        "mmlu_pro_economics", "mmlu_pro_engineering",
        "mmlu_pro_health", "mmlu_pro_history", "mmlu_pro_law", 
        "mmlu_pro_math", "mmlu_pro_other", "mmlu_pro_philosophy", 
        "mmlu_pro_physics", "mmlu_pro_psychology"
    ]
    
    tasks = [
        # "math",
        # "mt-bench",
        "mmlu_pro_biology", "mmlu_pro_business",
        "mmlu_pro_chemistry", "mmlu_pro_computer_science",
        "mmlu_pro_economics", "mmlu_pro_engineering",
        "mmlu_pro_health", "mmlu_pro_history", "mmlu_pro_law", 
        "mmlu_pro_math", "mmlu_pro_other", "mmlu_pro_philosophy", 
        "mmlu_pro_physics", "mmlu_pro_psychology",
        # "magpie_5k_test",  
        # "alpaca_5k_test",
        # # "metamath_5k_test",
        # "numina_cot_5k_test",
        # "mmlu_test",

    ]  

    train_tasks =["alpaca_5k_train","numina_cot_5k_train","mmlu_train"]
    
    probe_type = ["hs_last_mlp","mean","coe_dual_mlp"]
    def _build_hs_path(task: str):
        base_dir = os.path.join("..", "hs")
        if task.startswith("mmlu_pro_"):
            base_dir = os.path.join(base_dir, "mmlu_pro")
        return os.path.join(base_dir, f"Llama-3.1-8B-Instruct_{task}.pt")
        # return os.path.join(base_dir, f"Qwen2.5-1.5B-Instruct_{task}.pt")


    if mode == "get_scores":
        for task in ["magpie_5k_test","alpaca_5k_test","alpaca_5k_train"]:
            score = get_task_score(config, task=task)
            print(f"{task}: {score}")
    
    elif mode == "get_logits":
        for task in ["dapo-math-17k_dedup"]:
            task_path = os.path.join("./results",f"{task}.jsonl")
            if task.startswith("mmlu_pro_"):
                task_path = os.path.join("./results/mmlu_pro",f"{task}.jsonl")
            generate_logits(config, task=task,task_path=task_path)  
    
    elif mode == "train":
        for type in probe_type:
            config.training.probe_save_path= "./probe_save/max_k"
            config.router.probe_type = type
            pipeline = RouterEvaluationPipeline(config)
            # complete_probe_training_pipeline_with_mixed_datasets(
            #     config, 
            #     mix_strategy="balanced",
            #     task_list = ["numina_cot_5k_train","mmlu_"]
            
            # )
        
            # 不同采样大小训练
            for i in [400]:
                complete_probe_training_pipeline_with_mixed_datasets(
                    config, task_list=train_tasks,
                    mix_strategy="balanced", max_samples=i
                )
            
        # 层级训练
        # complete_layerwise_probe_training_pipeline(
        #     config, task_list=["mmlu_train","numina_cot_5k_train","magpie_5k_train"],
        #     mix_strategy="balanced"
        # )
        
        
        # categories = {
        # "STEM": ["biology", "chemistry", "computer_science", "engineering", "math", "physics"],
        # "Humanities": ["history", "philosophy"],  
        # "Social Science": ["economics", "law", "psychology"],
        # "Other": ["business", "health", "other"]      
        # }
        
        # # 三缺一（MMLU-Pro 按大类留一训练）
        # print("\n================ LEAVE-ONE-CATEGORY (MMLU-Pro) TRAINING ================")
        # for leave_out in categories.keys():
        #     include_categories = [c for c in categories.keys() if c != leave_out]
        #     task_list = []
        #     for cat in include_categories:
        #         for subject in categories[cat]:
        #             task_list.append(f"mmlu_pro_{subject}")

        #     print("\n" + "=" * 80)
        #     print(f"🚀 Training with three categories, leaving out: {leave_out}")
        #     print(f"📚 Using subjects from: {', '.join(include_categories)}")
        #     print(f"🧩 Total tasks: {len(task_list)}")

        #     # Use short, clear checkpoint names in subdir probe_save/mmlu_pro
        #     save_subdir = "hs_last_mlp"
        #     custom_save_name = f"leaveout_{leave_out.lower()}"

        #     complete_probe_training_pipeline_with_mixed_datasets(
        #         config,
        #         task_list=task_list,
        #         mix_strategy="balanced",
        #         save_subdir=save_subdir,
        #         custom_save_name=custom_save_name,
        #     )
        print("\n✅ Completed leave-one-category training for all four choices.")
        
    # elif mode == "train_rm":
        # config.router.router_type="trained_deberta"
        # complete_reward_training_pipeline                
    elif mode == "eval_single":
        for task in tasks:
            hidden_states_file = _build_hs_path(task)

            datasets = [f"{task}"]
            results = pipeline.evaluate_complete_pipeline(
                hidden_states_file=hidden_states_file,
                datasets=datasets
            )
     
    elif mode == "eval_base":
        probe_dir = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/base"
        probe_files = sorted(glob.glob(f"{probe_dir}/*.pt"))
        probe_configs = []
        
        for pf in probe_files:

            m = re.search(r'cot_5k_train_(.+)\.pt$', pf)
            
            if m:
                probe_type = m.group(1)  # 提取 train 后面的所有内容作为 probe_type
                metric_results_dir = f"metric_results/weighted/{probe_type}"
                
                probe_configs.append({
                    "checkpoint_path": pf,
                    "probe_type": probe_type,
                    "metric_results_dir": metric_results_dir
                })
        batch_evaluate_probes(config, probe_configs, tasks)    

    elif mode == "eval_max_k":  #data_size
        #统一一下两类方法命名格式
        probe_dir = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save/max_k"
        # probe_dir = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save_dynamic/max_k"
        probe_files = sorted(glob.glob(f"{probe_dir}/*.pt"))
        
        print(f"Found {len(probe_files)} probe files in {probe_dir}")
        
        probe_configs = []
        for pf in probe_files:
            filename = os.path.basename(pf)
            

            # sample_match = re.search(r'4000samples_probe_(\w+)\.pt$', filename)
            # sample_match = re.search(r'cot_5k_train_(\w+)_probe\.pt$', filename)
            # sample_match = re.search(r'balanced_probe_(\w+)\.pt$', filename)
            # sample_match = re.search(r'_train_(\w+)_probe\.pt$', filename)

            sample_match = re.search(r'_cot_5k_train_(\w+)_(\d+k)\.pt$', filename)

            if sample_match:
                base_probe = sample_match.group(1)  # 得到 "coe_dual"
                samples = sample_match.group(2)     # 得到 "1k", "4k" 等
                probe_type = base_probe 
                print

                
                            
                metric_results_dir = f"metric_results/max_k/{samples}_{probe_type}"
                
                
                probe_configs.append({
                    "checkpoint_path": pf,
                    "probe_type": probe_type,
                    "metric_results_dir": metric_results_dir,
                            })
        batch_evaluate_probes(config, probe_configs, tasks)
     
    elif mode == "eval_batch_mmlu_pro":
        # mmlu_pro leave-one-out
        probe_root = "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src/probe_save_dynamic/mmlu_pro"
        subdirs = [d for d in os.listdir(probe_root) if os.path.isdir(os.path.join(probe_root, d))]

        probe_configs = []
        for sub in sorted(subdirs):
            sub_dir = os.path.join(probe_root, sub)
            for pf in sorted(glob.glob(f"{sub_dir}/*.pt")):
                base = os.path.splitext(os.path.basename(pf))[0]
                # Use absolute path for metric results to avoid cwd/parents issues
                metric_results_dir = os.path.join(
                    "/volume/pt-train/users/wzhang/ghchen/zh/CoBench/src",
                    "metric_results",
                    "mmlu_pro",
                    sub,
                    base,
                )
                probe_configs.append({
                    "checkpoint_path": pf,
                    "probe_type": sub,
                    "metric_results_dir": metric_results_dir,
                })

        print(f"Discovered {len(probe_configs)} probes under {probe_root}")

        batch_evaluate_probes(config, probe_configs, mmlu_pro_tasks)

    elif mode == "semantic_entropy":
        config.metric_results_dir = "metric_results/weighted/semantic_entropy"
        config.router.router_type = "semantic_entropy"
        config.router.model_path = None 
        config.router.num_samples = 5  # semantic entropy计算的样本数
        
        pipeline = RouterEvaluationPipeline(config)
              
        results = {}
        
        # 评估每个任务
        for i, task in enumerate(tasks):
            print(f"\n{'='*60}")
            hidden_states_file = _build_hs_path(task)
            

            datasets = [task]
            pipeline.evaluate_complete_pipeline(
                hidden_states_file=hidden_states_file,
                datasets=datasets
            )

    elif mode == "self_questioning":
  
        config.metric_results_dir = "metric_results/weighted/self_questioning"
        config.router.router_type = "self_questioning"
        config.router.model_path = None  
        config.router.num_samples = 8  # self-ask计算的样本数
        
        pipeline = RouterEvaluationPipeline(config)
        
        
        results = {}
        
        # 评估每个任务
        for i, task in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"评估任务 {i+1}/{len(tasks)}: {task}")
            print(f"{'='*60}")

            hidden_states_file = _build_hs_path(task)
            
           
            datasets = [task]
            task_results = pipeline.evaluate_complete_pipeline(
                    hidden_states_file=hidden_states_file,
                    datasets=datasets
            )
                
    
    elif mode == "logits_based_routers":
        """Test logits-based routers"""
        from router import RouterManager

        # Create router manager and register logits-based routers
        router_manager = RouterManager()
        router_manager.create_max_logits_router()
        router_manager.create_top10_variance_router()
        router_manager.create_coe_router()
        router_manager.create_entropy_router()
        router_manager.create_confidence_margin_router()

      
        router_types = ["max_logits", "top10_variance", "coe", "entropy", "confidence_margin"]

        for router_type in router_types:
            print(f"\n{'='*60}")
            print(f"🚀 Testing Router: {router_type}")
            print(f"{'='*60}")

            router_results = {}

            for task in tasks:
                print(f"\nEvaluating task: {task}")

                hidden_states_file = _build_hs_path(task)

                if not os.path.exists(hidden_states_file):
                    print(f"Warning: Hidden states file not found: {hidden_states_file}")
                    router_results[task] = {"error": "Hidden states file not found"}
                    continue

                # Create a config copy for this router
                config_copy = copy.deepcopy(config)
                config_copy.router.router_type = router_type
                config_copy.metric_results_dir = f"metric_results/weighted/{router_type}"

                # Create pipeline with the specific router config
                pipeline_test = RouterEvaluationPipeline(config_copy)

                datasets = [task]
                pipeline_test.evaluate_complete_pipeline(
                        hidden_states_file=hidden_states_file,
                        datasets=datasets
                    )

    else:
        print(f"❌ 未知模式: {mode}")
        print("可用模式: train, eval_single, eval_batch, eval_batch_max_k, eval_batch_mmlu_pro, mt-bench, get_hs, logits_margin, semantic_entropy, self-ask, logits_based_routers")