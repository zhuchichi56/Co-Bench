import asyncio
import httpx
from typing import List
from loguru import logger
import re
from transformers import AutoTokenizer

from regex import T
from inference.gpt_inference import parallel_inference_gpt

def parser_score(input_list: List[str]) -> List[int]:
    pattern = re.compile(r'score:\s*(\d)', re.IGNORECASE)
    scores = [int(match.group(1)) if (match := pattern.search(s)) else 0 for s in input_list]
    return scores


def get_template(prompt, template_type="default", tokenizer=None):
    # logger.info(f"Using template type: {template_type}")
    if template_type == "alpaca":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    elif template_type == "tags":
        return f"""You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{"tag": "str", "explanation": "str"}}.
Query: {prompt} 
Assistant:"""
    elif template_type == "direct":
        return prompt
    else:
            # messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        messages = [
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


async def distribute_requests(prompt_list: List[str], 
                              max_tokens: int = 256, 
                              temperature: float = 0.0, 
                              top_p: float = 0.9, 
                              skip_special_tokens: bool = True, 
                              score = False, 
                              servers: List[str] = None,
                              template_type: str = "default",
                              tokenizer: str = None
                              ) -> List[str]:
    
    prompt_list = [get_template(prompt, template_type=template_type, tokenizer=tokenizer) for prompt in prompt_list]
    logger.info(f"Prompt list's first 1 element: {prompt_list[0]}")
    
    n_chunks = len(servers)
    chunk_size = len(prompt_list) // n_chunks
    chunks = [prompt_list[i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks)]
    
    if len(prompt_list) % n_chunks != 0:
        chunks[-1].extend(prompt_list[n_chunks * chunk_size:])
    
    for i in range(n_chunks):
        logger.info(f"Chunk {i} size: {len(chunks[i])}")
    
    
    tasks = [fetch_results(servers[i], chunks[i], max_tokens, temperature, top_p, skip_special_tokens) for i in range(n_chunks)]
   
    results = await asyncio.gather(*tasks)
    results = sum(results, [])
    return results if not score else parser_score(results)

async def fetch_results(server_url: str, 
                        chunk: List[str], 
                        max_tokens: int, 
                        temperature: float, 
                        top_p: float, 
                        skip_special_tokens: bool):
    async with httpx.AsyncClient(timeout=3600.0) as client:  # 将超时时间设置为30秒
        response = await client.post(f"{server_url}/inference", json={
            "input_data": chunk,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "skip_special_tokens": skip_special_tokens
        })
        response.raise_for_status()
        return response.json()["outputs"]
       


def parallel_inference(prompt_list: List[str],
                       max_tokens: int = 256,
                       temperature: float = 0.0,
                       top_p: float = 0.9,
                       template_type="default",
                       model_name_or_path: str = None,
                       type: str = "strong") -> List[str]:
    # Check if this is a GPT model request

    if model_name_or_path and "gpt" in model_name_or_path.lower():
        # Route to GPT inference
        import tempfile
        import os

        # Create a temporary output file for GPT inference
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        temp_file.close()

        try:
            gpt_model = (
                "gpt-5" if "gpt-5" in model_name_or_path.lower() else
                "gpt-4o" if "gpt-4o" in model_name_or_path.lower() else
                "gpt-4o" if "gpt-4" in model_name_or_path.lower() else
                "gpt-3.5-turbo" if "gpt-3.5" in model_name_or_path.lower() else
                "gpt-4o"
            )
            logger.info(f"Routing to GPT inference with model: {gpt_model}")

            results = parallel_inference_gpt(
                queries=prompt_list,
                output_file=temp_file.name,
                model=gpt_model,
                temperature=temperature,
                # top_p=top_p,
                max_tokens=max_tokens,
                system_prompt="You are a helpful AI assistant.",
                max_workers=32,
                batch_size=8
            )

            return results

        finally:
            # Clean up temp file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    else:
        # Original VLLM inference
        if type == "strong":
            gpu_ids = [2,3,4,5]
        else:  # weak
            gpu_ids = [2,3,4,5]
        print(gpu_ids)
        servers = ["http://localhost:800{}".format(i) for i in gpu_ids]
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return asyncio.run(distribute_requests(prompt_list,
                                                max_tokens,
                                                temperature,
                                                top_p,
                                                template_type=template_type,
                                                skip_special_tokens=True,
                                                servers=servers,
                                                tokenizer=tokenizer))


def parallel_inference_instagger(
    prompt_list: List[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    model_name_or_path: str = None
) -> List[str]:
    gpu_ids = [2,3,4,5]
    servers = ["http://localhost:800{}".format(i) for i in gpu_ids]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return asyncio.run(
        distribute_requests(
            prompt_list,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            skip_special_tokens=True,
            servers=servers,
            template_type="tags",
            tokenizer=tokenizer
        )
    )



if __name__ == "__main__":
    # Test parallel inference
    test_prompts = [
        "Tell me about cats.",
        "What is the capital of France?",
        "Explain quantum physics.",
        "Write a haiku about spring."
    ]
    
    print("Testing parallel inference...")
    results = parallel_inference(
        test_prompts,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        model_name_or_path="/volume/pt-train/models/Llama-3.1-8B-Instruct"
    )#、
    
    print("\nResults:")
    for prompt, result in zip(test_prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")

# python start.py --model_path /home/zhe/models/lukeminglkm/instagger_llama2 --base_port 8000 --gpu_list 1,2,3,4,5,6,7




