import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from loguru import logger
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
def load_finished_queries(output_file: str) -> set:
    """Load already processed queries from output file."""
    finished = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "query_id" in data:
                        finished.add(data["query_id"])
                except Exception as e:
                    logger.warning(f"Failed to parse line: {e}")
    return finished

def load_responses_from_file(output_file: str, queries: List[str]) -> Optional[List[str]]:
    """If all responses for queries are already in the file, load and return them."""
    if not os.path.exists(output_file):
        return None
    responses = {}
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "query_id" in data and "response" in data:
                    responses[data["query_id"]] = data["response"]
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")
    # Check if all queries are present
    all_found = True
    result_list = []
    for idx, _ in enumerate(queries):
        query_id = str(idx + 1)
        if query_id in responses:
            result_list.append(responses[query_id])
        else:
            all_found = False
            break
    if all_found:
        logger.info(f"All {len(queries)} queries found in {output_file}, loading from file.")
        return result_list
    return None

def save_results_atomic(results: List[Dict], output_file: str, lock: threading.Lock):
    """Save results to file in a thread-safe way."""
    with lock:
        with open(output_file, "a", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

def batch_iter(lst, batch_size):
    """Yield successive batch_size-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def parallel_inference_gpt(
    queries: List[str],
    output_file: str,
    model: str = "gpt-5",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    system_prompt: str = "You are a helpful AI.",
    max_workers: int = 64,
    batch_size: int = 16,
    
    **kwargs
) -> List[str]:
    # If results are already stored in file, load and return them
    loaded = load_responses_from_file(output_file, queries)
    if loaded is not None:
        return loaded

    api_key = "sk-e9OcUwV80NLvl00o73E5F3FcFa2d4a6cB7D46cB3D6263a1a"
    base_url = "https://api.ai-gaochao.cn/v1"
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    client = OpenAI(api_key=api_key, base_url=base_url)
    lock = threading.Lock()

    # Load finished query_ids for resuming
    finished_ids = load_finished_queries(output_file)
    logger.info(f"Loaded {len(finished_ids)} finished queries from {output_file}")

    # Prepare queries with query_id and messages
    all_queries = []
    for idx, user_query in enumerate(queries):
        query_id = str(idx + 1)
        if query_id in finished_ids:
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        all_queries.append({
            "query_id": query_id,
            "messages": messages,
            "instruction": user_query
        })
    logger.info(f"Pending queries: {len(all_queries)}")

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
            result = {
                "query_id": query["query_id"],
                "instruction": query["instruction"],
                "response": response.choices[0].message.content,
            }
            return result
        except Exception as e:
            logger.error(f"Error for query_id {query.get('query_id')}: {e}")
            return None

    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in tqdm(list(batch_iter(all_queries, batch_size)), desc="Batches"):
            futures = [executor.submit(infer_one, q) for q in batch]
            results = []
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    results.append(res)
            if results:
                save_results_atomic(results, output_file, lock)
                logger.info(f"Saved {len(results)} results to {output_file}")
                for item in results:
                    all_results.append(item["response"])

    # After inference, try to load all responses from file again (in case of resume)
    loaded = load_responses_from_file(output_file, queries)
    if loaded is not None:
        return loaded
    # Fallback: return what we have
    return all_results


# TODO： Usage
if __name__ == "__main__":
    OUTPUT_FILE = "model_results/gpt4o_math.jsonl"
    queries = pd.read_json("/HOME/sustc_ghchen/sustc_ghchen_4/CoBench/CoBench/src/data/math.jsonl",lines = True)
    queries = queries["instruction"]
    SYSTEM_PROMPT = "You are a helpful assistant that identifies paper names and abbreviations."

    logger.info("Starting DeepSeek API batch inference")
    results = parallel_inference_gpt(
        queries=queries,
        output_file=OUTPUT_FILE,
        model="gpt5",
        system_prompt=SYSTEM_PROMPT,
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        stop=None,
        max_workers=32,
        batch_size=8
    )
    print(results)