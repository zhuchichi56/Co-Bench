import random
import json
import os
import asyncio
import time
from tqdm import tqdm
from dataclasses import dataclass, asdict
from smolagents import OpenAIServerModel
from smolagents import CodeAgent, WebSearchTool, DuckDuckGoSearchTool
from smolagents.monitoring import LogLevel
from smolagents.utils import make_json_serializable

@dataclass
class Runs:
    step_number: int
    final_answer: str|float|int|None
    error_info: str|None
    full_steps: list
    memory_messages: list

@dataclass
class Example:
    id: str
    question: str
    answer: str
    model_id: str
    max_steps: int
    runs: list[Runs]


def load_data(dataset_name):
    import json
    import requests

    if dataset_name == "ASearcherBase35k":
        url = "https://huggingface.co/datasets/inclusionAI/ASearcher-train-data/resolve/main/ASearcher-Base-35k.jsonl"
    elif dataset_name == "ASearcherLRM35k":
        url = "https://huggingface.co/datasets/inclusionAI/ASearcher-train-data/resolve/main/ASearcher-LRM-35k.jsonl"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    response = requests.get(url)
    response.raise_for_status()

    data = []
    for line in response.text.strip().split('\n'):
        if line.strip():
            item = json.loads(line)
            if dataset_name == "ASearcherBase35k":
                item["id"] = str(item["qid"])
                item["answer"] = item["answer"][0]
            data.append(item)

    return data


def process_single_question(question, model_id, max_steps=20, max_retries=2):
    for retry_count in range(max_retries + 1):
        model = OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:8000/v1",
            api_key="empty",
        )

        agent = CodeAgent(
            tools=[DuckDuckGoSearchTool()],
            model=model,
            stream_outputs=False,
            add_base_tools=True,
            additional_authorized_imports=[],
            use_structured_outputs_internally=True,
            verbosity_level=LogLevel.OFF,
        )

        try:
            agent.run(question, max_steps=max_steps)
            return agent
        except Exception as e:
            error_str = str(e).lower()
            if "Rate limit" in error_str and retry_count < max_retries:
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

                    # Check for Azure CLI errors and skip adding to processed_ids to allow regeneration
                    error_info = example['runs'][0]['error_info']
                    if error_info and ("Azure CLI" in error_info or "az login" in error_info or "Rate limit" in error_info):
                        print(f"{count}. ID: {example['id']} - Azure CLI error detected, will regenerate\n")
                        continue

                    processed_ids.add(example['id'])
                    print(f"{count}. ID: {example['id']}, Question: {example['question'][:50]}...")
                    print(f"   Answer: {example['answer']}")
                    if example['runs'][0]['final_answer']:
                        answer_str = str(example['runs'][0]['final_answer'])
                        print(f"   Output: {answer_str}")
                    else:
                        print("   Output: None")
                        if example['runs'][0]['error_info']:
                            print(f"   Error Info: {example['runs'][0]['error_info']}")
                    print(f"   Steps: {example['runs'][0]['step_number']}")
                    print()
        print(f"Processed data count: {count}")
    else:
        print("File does not exist")

    # Return remaining data items that need to be processed
    remaining_data = [item for item in all_data if item['id'] not in processed_ids]
    print(f"Remaining data to process: {len(remaining_data)}")
    return remaining_data


async def process_single_item(item, semaphore, model_id, n_runs, max_steps):
    async with semaphore:
        qid, question, true_answer = item["id"], item["question"], item["answer"]

        # Run multiple times and collect all runs
        all_runs = []
        for _ in range(n_runs):
            # Run in thread pool to avoid blocking
            agent = await asyncio.get_event_loop().run_in_executor(
                None, process_single_question, question, model_id, max_steps
            )
            run_data = pack_single_run(agent)
            all_runs.append(run_data)

            # Output current run result using tqdm.write to avoid interfering with progress bar
            tqdm.write(f"Q: {question[:100]}...")
            tqdm.write(f"A: {true_answer}")
            if run_data.final_answer:
                tqdm.write(f"Output: {str(run_data.final_answer)}")
            else:
                tqdm.write("Output: None")
                if run_data.error_info:
                    tqdm.write(f"Error: {run_data.error_info[:100]}")
            tqdm.write(f"Steps: {run_data.step_number}")
            tqdm.write("-" * 50)

        # Create Example object
        example = Example(
            id=qid,
            question=question,
            answer=true_answer,
            model_id=model_id,
            max_steps=max_steps,
            runs=all_runs
        )

        return {
            "id": qid,
            "example": asdict(example)
        }


async def process_all_questions(inputs, model_id, output_file, concurrent_limit=10, n_runs=1, max_steps=20):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    semaphore = asyncio.Semaphore(concurrent_limit)
    # Create output file in append mode with proper async handling
    async with asyncio.Lock():
        with open(output_file, 'a', encoding='utf-8') as f:
            pass  # Just ensure file exists

    tasks = [process_single_item(input, semaphore, model_id, n_runs, max_steps) for input in inputs]

    # Process with real-time progress updates
    with open(output_file, 'a', encoding='utf-8') as f:
        for task in tqdm(asyncio.as_completed(tasks), total=len(inputs), desc="Processing questions"):
            result = await task
            json.dump(make_json_serializable(result["example"]), f, ensure_ascii=False)
            f.write('\n')
            f.flush()  # Ensure immediate write

            # Check for Azure CLI errors and exit if found
            error_info = result["example"]["runs"][0]["error_info"]
            if error_info and ("Azure CLI" in error_info or "az login" in error_info):
                print("Detected Azure CLI error, stopping execution")
                return


async def main():
    dataset = "ASearcherLRM35k"
    # dataset = "ASearcherBase35k"

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    # model_id = "Qwen/Qwen3-8B-Base"
    # model_id = "Qwen/Qwen3-8B"

    n_samples = -1
    n_runs = 1
    max_steps = 20
    concurrent_limit = 10
    output_file = f"outputs/code_agent/asearcher/{dataset}_{model_id.split('/')[-1]}.jsonl"
    # output_file = f"outputs/code_agent/asearcher/{dataset}_{model_id.split('/')[-1]}-think.jsonl"
    # output_file = f"outputs/code_agent/asearcher/{dataset}_{model_id.split('/')[-1]}-no-think.jsonl"

    data = load_data(dataset)

    # Apply n_samples limit to total data first
    if n_samples > 0:
        data = data[:n_samples]

    # Get remaining data to process
    remaining_data = get_remaining_data(output_file, data)

    await process_all_questions(
        inputs=remaining_data,
        model_id=model_id,
        output_file=output_file,
        concurrent_limit=concurrent_limit,
        n_runs=n_runs,
        max_steps=max_steps
    )

if __name__ == "__main__":
    asyncio.run(main())


"""
screen -S vllm
source uv_smolagents/bin/activate
vllm serve Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable_prefix_caching

vllm serve Qwen/Qwen3-8B-Base \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable_prefix_caching

vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable_prefix_caching

vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable_prefix_caching \
    --chat-template ./qwen3_nonthinking.jinja
"""