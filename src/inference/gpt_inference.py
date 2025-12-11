import os
import json
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from openai import OpenAI
from loguru import logger
from tqdm import tqdm
from config import PipelineConfig

# Azure imports are lazy-loaded to avoid dependency errors when not using Azure


def load_finished_queries(output_file: str) -> Dict[str, str]:
    """Load completed query results from output file."""
    completed = {}
    if not os.path.exists(output_file):
        return completed
    
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "query_id" in data and "response" in data:
                    completed[data["query_id"]] = data["response"]
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")
    
    return completed


def load_responses_from_file(output_file: str, queries: List[str]) -> Optional[List[str]]:
    """Check if all responses exist and return them in order."""
    completed = load_finished_queries(output_file)
    
    if len(completed) < len(queries):
        return None
    
    result = []
    for idx in range(len(queries)):
        query_id = str(idx + 1)
        if query_id not in completed:
            return None
        result.append(completed[query_id])
    
    logger.info(f"All {len(queries)} queries found in {output_file}, loading from file.")
    return result


def save_result_atomic(result: Dict, output_file: str, lock: threading.Lock):
    """Save single result in a thread-safe way."""
    with lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def save_result_atomic_async(result: Dict, output_file: str, lock: asyncio.Lock):
    """Save single result in an async-safe way."""
    async with lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _parallel_inference_gpt_openai(
    queries: List[str],
    output_file: str,
    model: str = "gpt-5",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    max_workers: Optional[int] = None,
    batch_size: int = 16,  # Keep parameter but not used
    inference_config=None,
    **kwargs
) -> List[str]:
    
    resolved_config = inference_config or PipelineConfig.from_yaml().inference
    
    # Check if all results already exist
    loaded = load_responses_from_file(output_file, queries)
    if loaded is not None:
        return loaded

    api_key = resolved_config.openai_api_key or os.getenv("OPENAI_API_KEY")
    base_url = resolved_config.openai_api_base or os.getenv("OPENAI_API_BASE") or "https://api.ai-gaochao.cn/v1"
    resolved_max_workers = max_workers if max_workers is not None else resolved_config.max_workers
    resolved_system_prompt = system_prompt if system_prompt is not None else resolved_config.system_prompt

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    lock = threading.Lock()

    # Load finished queries for resuming
    finished = load_finished_queries(output_file)
    logger.info(f"Loaded {len(finished)} finished queries from {output_file}")

    # Prepare pending queries
    pending_queries = []
    for idx, user_query in enumerate(queries):
        query_id = str(idx + 1)
        if query_id in finished:
            continue
        messages = [
            {"role": "system", "content": resolved_system_prompt},
            {"role": "user", "content": user_query}
        ]
        pending_queries.append({
            "query_id": query_id,
            "messages": messages,
            "instruction": user_query
        })
    
    logger.info(f"Pending queries: {len(pending_queries)}")

    def infer_one(query: Dict[str, Any]) -> Optional[Dict]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=query["messages"],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs
            )
            content = response.choices[0].message.content
           
            return {
                "query_id": query["query_id"],
                "instruction": query["instruction"],
                "response": content,
            }
        except Exception as e:
            logger.error(f"Error for query_id {query.get('query_id')}: {e}")
            return None

    # Submit all tasks at once
    with ThreadPoolExecutor(max_workers=resolved_max_workers) as executor:
        futures = {
            executor.submit(infer_one, q): q["query_id"] 
            for q in pending_queries
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                save_result_atomic(result, output_file, lock)

   

    # Load all results in order
    loaded = load_responses_from_file(output_file, queries)
    if loaded is not None:
        return loaded
    
    # Fallback: return what we have
    completed = load_finished_queries(output_file)
    return [completed.get(str(i + 1), "") for i in range(len(queries))]


def _parallel_inference_gpt_azure(
    queries: List[str],
    output_file: str,
    model: str = "gpt-5",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    max_workers: Optional[int] = None,
    batch_size: int = 16,
    inference_config=None,
    **kwargs
) -> List[str]:
    """Azure OpenAI implementation using async methods."""

    # Lazy import Azure dependencies
    try:
        from .gpt_inference_azure import get_client, build_messages, get_response
        from azure.identity import AzureCliCredential
    except ImportError as e:
        logger.error(f"Azure dependencies not installed. Please run: pip install azure-identity")
        raise ImportError(
            "Azure OpenAI backend requires azure-identity package. "
            "Install it with: pip install azure-identity"
        ) from e

    resolved_config = inference_config or PipelineConfig.from_yaml().inference

    # Check if all results already exist
    loaded = load_responses_from_file(output_file, queries)
    if loaded is not None:
        return loaded

    resolved_system_prompt = system_prompt if system_prompt is not None else resolved_config.system_prompt
    concurrency = max_workers if max_workers is not None else resolved_config.max_workers

    # Load finished queries for resuming
    finished = load_finished_queries(output_file)
    logger.info(f"Loaded {len(finished)} finished queries from {output_file}")

    # Prepare pending queries
    pending_queries = []
    for idx, user_query in enumerate(queries):
        query_id = str(idx + 1)
        if query_id in finished:
            continue
        pending_queries.append({
            "query_id": query_id,
            "instruction": user_query,
        })

    logger.info(f"Pending queries: {len(pending_queries)}")

    if not pending_queries:
        logger.info("All queries already processed.")
        return load_responses_from_file(output_file, queries) or []

    async def infer_one_azure(
        query: Dict[str, Any],
        credential: AzureCliCredential,
        semaphore: asyncio.Semaphore,
        write_lock: asyncio.Lock,
        bar: Optional[tqdm]
    ) -> Optional[Dict]:
        async with semaphore:
            try:
                client, resolved_model = get_client(model_name=model, credential=credential)
                messages = build_messages(query["instruction"], resolved_system_prompt)

                # For GPT-5, don't pass max_tokens and temperature
                if "gpt-5" in resolved_model or "gpt-5" in model:
                    response_text = await asyncio.to_thread(
                        get_response,
                        client,
                        resolved_model,
                        messages,
                    )
                else:
                    response_text = await asyncio.to_thread(
                        get_response,
                        client,
                        resolved_model,
                        messages,
                        max_tokens,
                        temperature,
                        top_p,
                    )

                result = {
                    "query_id": query["query_id"],
                    "instruction": query["instruction"],
                    "response": response_text,
                }

                # Save result atomically
                await save_result_atomic_async(result, output_file, write_lock)

                if bar:
                    bar.update(1)

                return result

            except Exception as e:
                logger.error(f"Error for query_id {query.get('query_id')}: {e}")
                if bar:
                    bar.update(1)
                return None

    async def run_all_azure():
        credential = AzureCliCredential()
        semaphore = asyncio.Semaphore(min(concurrency, len(pending_queries)))
        write_lock = asyncio.Lock()

        bar = tqdm(total=len(pending_queries), desc="Processing", unit="item")

        tasks = [
            asyncio.create_task(
                infer_one_azure(q, credential, semaphore, write_lock, bar)
            )
            for q in pending_queries
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            if bar:
                bar.close()

    # Run async tasks
    asyncio.run(run_all_azure())

    # Load all results in order
    loaded = load_responses_from_file(output_file, queries)
    if loaded is not None:
        return loaded

    # Fallback: return what we have
    completed = load_finished_queries(output_file)
    return [completed.get(str(i + 1), "") for i in range(len(queries))]


def parallel_inference_gpt(
    queries: List[str],
    output_file: str,
    model: str = "gpt-5",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None,
    max_workers: Optional[int] = None,
    batch_size: int = 16,
    inference_config=None,
    use_azure: Optional[bool] = None,
    **kwargs
) -> List[str]:
    """
    Unified GPT inference function that routes to OpenAI or Azure based on config.

    Args:
        queries: List of user queries to process
        output_file: Path to save results
        model: Model name (e.g., "gpt-5", "gpt-4o")
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens in response
        system_prompt: System prompt for the model
        max_workers: Maximum concurrent workers
        batch_size: Batch size (kept for compatibility)
        inference_config: Optional config object
        use_azure: If True, use Azure; if False, use OpenAI; if None, use config default
        **kwargs: Additional arguments

    Returns:
        List of response strings in same order as queries
    """
    resolved_config = inference_config or PipelineConfig.from_yaml().inference

    # Determine which backend to use
    should_use_azure = use_azure if use_azure is not None else getattr(resolved_config, 'use_azure', False)
    if should_use_azure:
        logger.info("Using Azure OpenAI backend")
        return _parallel_inference_gpt_azure(
            queries=queries,
            output_file=output_file,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            max_workers=max_workers,
            batch_size=batch_size,
            inference_config=inference_config,
            **kwargs
        )
    else:
        logger.info("Using OpenAI backend")
        return _parallel_inference_gpt_openai(
            queries=queries,
            output_file=output_file,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            max_workers=max_workers,
            batch_size=batch_size,
            inference_config=inference_config,
            **kwargs
        )


if __name__ == "__main__":
    OUTPUT_FILE = "results/gpt4o_math.jsonl"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    queries = ["How are you?", "Who are you?"] * 200
    SYSTEM_PROMPT = "You are a helpful assistant."

    logger.info("Starting batch inference")
    results = parallel_inference_gpt(
        queries=queries,
        output_file=OUTPUT_FILE,
        model="gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        max_workers=32
    )
    
    print(f"Completed {len(results)} inferences")
    
