"""
smolagents Agent 框架（带异步并发、断点续传、Template、Evaluate）
参考 ASearcher_vllm_async.py 的架构，整合原有的 template 和 evaluate 功能
"""
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

# 添加 xVerify 路径
PROJECT_ROOT = Path(__file__).parent.parent
XVERIFY_PATH = PROJECT_ROOT / "external" / "xverify"
sys.path.insert(0, str(XVERIFY_PATH))

from smolagents import (
    Tool, CodeAgent, TransformersModel, VLLMModel,
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
    """单次运行结果"""
    step_number: int
    final_answer: str | float | int | None
    error_info: str | None
    full_steps: list
    memory_messages: list


@dataclass
class Example:
    """完整的问题示例"""
    id: str
    instruction: str
    response: str  # 标准答案
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
            add_base_tools=True,
            additional_authorized_imports=[],
            use_structured_outputs_internally=True,
            verbosity_level=LogLevel.OFF,
        )

        try:
            # ✅ 应用模板
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
    """将agent执行结果打包成Runs对象"""
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
    """读取已处理数据，返回剩余待处理项"""
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
                        print(f"{count}. ID: {example['id']} - 检测到错误，将重新生成")
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
        print(f"已处理数据: {count}")
    else:
        print("输出文件不存在，将从头开始")

    # 返回剩余未处理的数据
    remaining_data = [item for item in all_data if item['id'] not in processed_ids]
    print(f"剩余待处理: {len(remaining_data)}")
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
    """异步处理单个数据项"""
    async with semaphore:
        qid, instruction, true_answer = item["id"], item["instruction"], item.get("response", "")

        # 运行多次并收集所有结果
        all_runs = []
        for _ in range(n_runs):
            # 在线程池中运行，避免阻塞
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

            # 输出当前运行结果（使用tqdm.write避免干扰进度条）
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

        # 创建Example对象
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
    """异步批量处理所有问题，保持原始顺序"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    semaphore = asyncio.Semaphore(concurrent_limit)

    # 确保输出文件存在
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

    results = []
    with tqdm(total=len(tasks), desc="Processing questions") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

            error_info = result["example"]["runs"][0]["error_info"]
            if error_info and ("Azure CLI" in error_info or "az login" in error_info):
                print("检测到Azure CLI错误，停止执行")
                results_sorted = sorted(results, key=lambda x: x["id"])
                with open(output_file, 'a', encoding='utf-8') as f:
                    for r in results_sorted:
                        json.dump(make_json_serializable(r["example"]), f, ensure_ascii=False)
                        f.write('\n')
                return

    results_sorted = sorted(results, key=lambda x: x["id"])
    with open(output_file, 'a', encoding='utf-8') as f:
        for result in results_sorted:
            json.dump(make_json_serializable(result["example"]), f, ensure_ascii=False)
            f.write('\n')
            f.flush()

async def main_async():
    parser = argparse.ArgumentParser(description="Agent 推理与评估（异步版本）")
    parser.add_argument("--agent_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--agent_tools", type=str, nargs="+", default=None)
    parser.add_argument("--template_type", type=str, default="default", choices=["default", "alpaca", "tags", "direct"])
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--n_runs", type=int, default=1, help="每个问题运行次数")
    parser.add_argument("--concurrent_limit", type=int, default=10, help="并发限制")
    parser.add_argument("--use_openai_server", action="store_true", help="使用OpenAI Server模式")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")

    args = parser.parse_args()

    # 处理工具参数
    tools = []
    if args.agent_tools:
        tool_mapping = {
            "DuckDuckGoSearchTool": DuckDuckGoSearchTool,
            "VisitWebpageTool": VisitWebpageTool,
        }
        for tool_name in args.agent_tools:
            if tool_name in tool_mapping:
                tools.append(tool_mapping[tool_name]())
                print(f"✅ 已加载工具: {tool_name}")
            else:
                print(f"⚠️  未知工具: {tool_name}")

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
        print(f"\n✅ 推理完成！")
    else:
        print("所有数据已处理完成！")


    complete_results = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                complete_results.append(json.loads(line))

    if not complete_results:
        print("没有可评估的数据！")
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
    """同步主函数入口"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


"""
使用示例：

1. 启动 vLLM 服务器：
screen -S vllm
vllm serve /volume/pt-train/models/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --enable_prefix_caching

2. 运行异步 Agent（带Template和Evaluate）：
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

3. 断点续传（自动跳过已处理的数据，继续评估）：
python src/agent.py --dataset gsm8k --use_openai_server --concurrent_limit 10
"""
