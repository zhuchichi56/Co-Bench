import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import argparse
import asyncio
import time
import os
from tqdm import tqdm
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent
XVERIFY_PATH = PROJECT_ROOT / "external" / "xverify"
sys.path.insert(0, str(XVERIFY_PATH))

from smolagents import (
    Tool, CodeAgent, VLLMModel,
    OpenAIServerModel, DuckDuckGoSearchTool, VisitWebpageTool
)
from smolagents.monitoring import LogLevel
from smolagents.utils import make_json_serializable
from transformers import AutoTokenizer

from data import DataLoader
from loss_calculator import MultiModalLossCalculator, llm_judge_general
from inference.vllm_client import get_template

@dataclass
class Runs:
    step_number: int
    final_answer: str | float | int | None
    error_info: str | None
    full_steps: list
    memory_messages: list


@dataclass
class Example:
    id: str
    instruction: str
    response: str
    model_id: str
    max_steps: int
    runs: list[Runs]
    dataset_type: str = "general"


def process_single_question(
    question: str,
    model_id: str,
    tools: List[Tool],
    template_type: str,
    tokenizer,
    max_steps: int = 20,
    max_retries: int = 2,
    use_openai_server: bool = False,
    api_base: str = "http://localhost:8000/v1"
):
    for retry_count in range(max_retries + 1):
        if use_openai_server:
            model = OpenAIServerModel(
                model_id=model_id,
                api_base=api_base,
                api_key="empty",
            )
        else:
            model = VLLMModel(
                model_id=model_id,
                model_kwargs={
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 4096,
                }
            )
            
        agent = CodeAgent(
            tools=tools,
            model=model,
            stream_outputs=False,
            use_structured_outputs_internally=True,
            verbosity_level=LogLevel.OFF,
        )

        try:
            if template_type == "search_guided":
                task_inference = f"""If you are uncertain, use web_search to find information.

{question}

Remember: For multiple choice questions, your final answer must be a single letter (A, B, C, D, or E)."""
            else:
                task_inference = get_template(
                    question,
                    template_type=template_type,
                    tokenizer=tokenizer
                )
            agent.run(task_inference, max_steps=max_steps)
            return agent
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str and retry_count < max_retries:
                wait_time = 5 * (retry_count + 1)
                time.sleep(wait_time)
                continue
            else:
                agent.step_number = 1
                agent.memory.steps = []
                agent._error_info = str(e)
                return agent

    return agent

def pack_single_run(agent):
    steps = agent.memory.steps

    try:
        full_steps = agent.memory.get_full_steps()
    except:
        full_steps = []

    try:
        memory_messages = agent.write_memory_to_messages()
    except:
        memory_messages = []

    if steps and hasattr(steps[-1], 'is_final_answer') and steps[-1].is_final_answer:
        final_answer = steps[-1].action_output
    else:
        final_answer = None

    error_info = None
    if hasattr(agent, '_error_info'):
        error_info = agent._error_info
    elif steps and hasattr(steps[-1], 'error') and steps[-1].error:
        error_info = steps[-1].error.message

    return Runs(
        step_number=agent.step_number,
        final_answer=final_answer,
        error_info=error_info,
        full_steps=full_steps,
        memory_messages=memory_messages
    )


def get_remaining_data(output_file, all_data):
    processed_ids = set()
    if os.path.exists(output_file):
        count = 0
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
                    example = json.loads(line)

                    error_info = example['runs'][0]['error_info']
                    if error_info and ("Azure CLI" in error_info or "az login" in error_info or "Rate limit" in error_info):
                        print(f"{count}. ID: {example['id']} - detected error; will re-generate")
                        continue

                    processed_ids.add(example['id'])
                    print(f"{count}. ID: {example['id']}, Instruction: {example['instruction'][:50]}...")
                    print(f"   Answer: {example['response']}")
                    if example['runs'][0]['final_answer']:
                        answer_str = str(example['runs'][0]['final_answer'])
                        print(f"   Output: {answer_str[:100]}")
                    else:
                        print("   Output: None")
                        if example['runs'][0]['error_info']:
                            print(f"   Error: {example['runs'][0]['error_info'][:100]}")
                    print(f"   Steps: {example['runs'][0]['step_number']}")
                    print()
        print(f"Processed data: {count}")
    else:
        print("Output file does not exist, starting from the beginning")

    remaining_data = [item for item in all_data if item['id'] not in processed_ids]
    print(f"Remaining to process: {len(remaining_data)}")
    return remaining_data

async def process_single_item(
    item,
    semaphore,
    model_id,
    tools,
    template_type,
    tokenizer,
    n_runs,
    max_steps,
    dataset_type,
    use_openai_server,
    api_base
):
    async with semaphore:
        qid, instruction, true_answer = item["id"], item["instruction"], item.get("response", "")

        all_runs = []
        for _ in range(n_runs):
            agent = await asyncio.get_event_loop().run_in_executor(
                None,
                process_single_question,
                instruction,
                model_id,
                tools,
                template_type,
                tokenizer,
                max_steps,
                2,  # max_retries
                use_openai_server,
                api_base
            )
            run_data = pack_single_run(agent)
            all_runs.append(run_data)

            tqdm.write(f"Q: {instruction[:100]}...")
            tqdm.write(f"A: {true_answer}")
            if run_data.final_answer:
                tqdm.write(f"Output: {str(run_data.final_answer)[:100]}")
            else:
                tqdm.write("Output: None")
                if run_data.error_info:
                    tqdm.write(f"Error: {run_data.error_info[:100]}")
            tqdm.write(f"Steps: {run_data.step_number}")
            tqdm.write("-" * 50)

        example = Example(
            id=qid,
            instruction=instruction,
            response=true_answer,
            model_id=model_id,
            max_steps=max_steps,
            runs=all_runs,
            dataset_type=dataset_type
        )

        return {
            "id": qid,
            "example": asdict(example)
        }


async def process_all_questions(
    inputs,
    model_id,
    tools,
    template_type,
    tokenizer,
    output_file,
    dataset_type,
    concurrent_limit=10,
    n_runs=1,
    max_steps=20,
    use_openai_server=False,
    api_base="http://localhost:8000/v1"
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    semaphore = asyncio.Semaphore(concurrent_limit)

    async with asyncio.Lock():
        with open(output_file, 'a', encoding='utf-8') as f:
            pass

    tasks = [
        process_single_item(
            input_item, semaphore, model_id, tools, template_type, tokenizer,
            n_runs, max_steps, dataset_type, use_openai_server, api_base
        )
        for input_item in inputs
    ]

    id_to_index = {item["id"]: idx for idx, item in enumerate(inputs)}

    completed_results = {}
    next_to_save_idx = 0

    with tqdm(total=len(tasks), desc="Processing questions") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            result_id = result["id"]
            result_idx = id_to_index[result_id]
            completed_results[result_idx] = result
            pbar.update(1)

            error_info = result["example"]["runs"][0]["error_info"]
            if error_info and ("Azure CLI" in error_info or "az login" in error_info):
                print("Detected Azure CLI error, stopping execution")
                sorted_indices = sorted(completed_results.keys())
                with open(output_file, 'a', encoding='utf-8') as f:
                    for idx in sorted_indices:
                        json.dump(make_json_serializable(completed_results[idx]["example"]), f, ensure_ascii=False)
                        f.write('\n')
                return

            with open(output_file, 'a', encoding='utf-8') as f:
                while next_to_save_idx in completed_results:
                    result_to_save = completed_results.pop(next_to_save_idx)
                    json.dump(make_json_serializable(result_to_save["example"]), f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                    next_to_save_idx += 1

async def main_async():
    parser = argparse.ArgumentParser(description="Agent inference and evaluation (async version)")
    parser.add_argument("--agent_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--agent_tools", type=str, nargs="+", default=None)
    parser.add_argument("--template_type", type=str, default="default",
                       choices=["default", "alpaca", "tags", "direct", "search_guided"],
                       help="search_guided: guide the model to use search tool")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--n_runs", type=int, default=1, help="number of runs per question")
    parser.add_argument("--concurrent_limit", type=int, default=10, help="concurrent limit")
    parser.add_argument("--use_openai_server", action="store_true", help="use OpenAI Server mode")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")

    args = parser.parse_args()

    tools = []
    if args.agent_tools:
        tool_mapping = {
            "DuckDuckGoSearchTool": DuckDuckGoSearchTool,
            "VisitWebpageTool": VisitWebpageTool,
        }
        for tool_name in args.agent_tools:
            if tool_name in tool_mapping:
                tools.append(tool_mapping[tool_name]())
                print(f"✅ Loaded tool: {tool_name}")
            else:
                print(f"⚠️ Unknown tool: {tool_name}")

    data_loader = DataLoader()
    data, dataset_type = data_loader.load_dataset(args.dataset)
    for i, item in enumerate(data):
        if 'id' not in item:
            item['id'] = str(i)
    if args.num_samples and args.num_samples > 0:
        data = data[:args.num_samples]


    tokenizer = AutoTokenizer.from_pretrained(args.agent_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    agent_name_short = args.agent_name.split("/")[-1]
    output_dir = Path(__file__).parent.parent /"src"/"results" / "agent" / agent_name_short
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.dataset}.jsonl"


    remaining_data = get_remaining_data(str(output_file), data)

    if remaining_data:
        await process_all_questions(
            inputs=remaining_data,
            model_id=args.agent_name,
            tools=tools,
            template_type=args.template_type,
            tokenizer=tokenizer,
            output_file=str(output_file),
            dataset_type=dataset_type,
            concurrent_limit=args.concurrent_limit,
            n_runs=args.n_runs,
            max_steps=args.max_steps,
            use_openai_server=args.use_openai_server,
            api_base=args.api_base
        )
        print(f"\n✅ Inference completed!")
    else:
        print("All data processed!")


    complete_results = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                complete_results.append(json.loads(line))

    if not complete_results:
        print("No data to evaluate!")
        return

    loss_calc = MultiModalLossCalculator(model=None, tokenizer=None, device=None)
    scores = []

    if dataset_type == "general":

        questions = [{"instruction": item['instruction']} for item in complete_results]

        answers = [{"response": str(item['runs'][0]['final_answer']) if item['runs'][0]['final_answer'] else ""} for item in complete_results]
        ref_answers = [{"response": item['response']} for item in complete_results]
        scores = llm_judge_general(
            questions, answers, "gpt-4o", ref_answers,
            max_workers=args.max_workers
        )
    else:
        for item in complete_results:
            generated_response = str(item['runs'][0]['final_answer']) if item['runs'][0]['final_answer'] else ""
            is_correct = loss_calc._evaluate_response(
                generated_response, item['response'], item['instruction'], dataset_type
            )
            score = 1.0 if is_correct else 0.0
            scores.append(score)

    accuracy = sum(scores) / len(scores) if scores else 0.0
    correct = sum(scores)
    total = len(scores)

    print(f"   accuracy: {accuracy:.2%}")


    updated_results = []
    for result, score in zip(complete_results, scores):
        result['score'] = score
        updated_results.append(result)


    with open(output_file, 'w', encoding='utf-8') as f:
        for result in updated_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


"""
Example:

1. Start vLLM server:
screen -S vllm
vllm serve /volume/pt-train/models/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable_prefix_caching

2. Run async Agent (with Template and Evaluate):
python src/agent.py \
    --agent_name /volume/pt-train/models/Llama-3.1-8B-Instruct \
    --dataset gsm8k \
    --num_samples 100 \
    --agent_tools DuckDuckGoSearchTool \
    --template_type default \
    --max_steps 20 \
    --concurrent_limit 10 \
    --n_runs 1 \
    --use_openai_server \
    --api_base "http://localhost:8000/v1"

3. Resume from checkpoint (skip processed data and continue evaluation):
python src/agent.py --dataset hotpotqa_500 --use_openai_server --concurrent_limit 10
"""
