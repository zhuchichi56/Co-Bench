import token
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import ray
from vllm import LLM, SamplingParams
from loguru import logger
import sys
import torch
# Define the configuration table
config_table = {
    "llama2": {
        "max_model_len": 4096,
        "id2score": {29900: "0", 29896: "1"}
    },
    "llama3": {
        "max_model_len": 8192,
        "id2score": {15: "0", 16: "1"}
    },
    "mistral": {
        "max_model_len": 2000,
        "id2score": {28734: "0", 28740: "1"}
    },
    "qwen3": {
        "max_model_len": 32768,
        "id2score": {}
    }
}

# Request and Response models
class InferenceRequest(BaseModel):
    input_data: List[str]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    avoid_words: List[List[str]] = None
    

class InferenceResponse(BaseModel):
    outputs: List[str]
    
class LogprobsResponse(BaseModel):
    logprobs: List[str]
    

app = FastAPI()

def get_model_config(model_path: str):
    for key in config_table:
        if key in model_path.lower():
            logger.info(f"Using config for {key}")
            return key, config_table[key]
    logger.info(f"Unknown model type, using default config without max_model_len")
    return "unknown", {}


@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    # logger.info(f"Inference request: {request}")
    global llm
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            skip_special_tokens=request.skip_special_tokens
        )
        outputs = llm.generate(request.input_data, sampling_params)
        output_texts = [output.outputs[0].text for output in outputs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    return InferenceResponse(outputs=output_texts)

 
@app.post("/inference_logprobs", response_model=LogprobsResponse)
def inference_logprobs(request: InferenceRequest):
    # logger.info(f"Inference log probs request: {request}")
    global llm
    try:
        sampling_params = SamplingParams(
            max_tokens=2,
            logprobs=20
        )
        outputs = llm.generate(request.input_data, sampling_params)
        # print(len(outputs))
        # print(len(outputs[0].outputs))
        # print(outputs[0].outputs[0].logprobs)
        output_texts = [output.outputs[0].logprobs[0] for output in outputs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    return InferenceResponse(outputs=output_texts)



class AvoidTokensLogitsProcessor:
    def __init__(self, avoid_token_ids, penalty=1e6):
        self.avoid_token_ids = avoid_token_ids
        self.penalty = penalty

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        for token_id in self.avoid_token_ids:
            scores[token_id] -= self.penalty 
        return scores


def build_logits_processor(avoid_words: List[List[str]], tokenizer) -> List[AvoidTokensLogitsProcessor]:
    avoid_token_ids_list = [[tokenizer.convert_tokens_to_ids(word) for word in word_list] for word_list in avoid_words]
    logits_processors = []
    for avoid_token_ids in avoid_token_ids_list:
        logits_processors.append(AvoidTokensLogitsProcessor(avoid_token_ids))
    return logits_processors

@app.post("/inference_thinkdifferent", response_model=InferenceResponse)
def inference_thinkdifferent(request: InferenceRequest):
    # logger.info(f"Think different inference request: {request}")
    global llm
    global tokenizer
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            skip_special_tokens=request.skip_special_tokens,
            logits_processors=build_logits_processor(avoid_words=request.avoid_words, tokenizer=tokenizer)
        )
        outputs = llm.generate(request.input_data, sampling_params)
        output_texts = [output.outputs[0].text for output in outputs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    return InferenceResponse(outputs=output_texts)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python vllm_server.py <port> <model_path>")
        sys.exit(1)
    port = int(sys.argv[1])
    model_path = sys.argv[2]

    model_key, config = get_model_config(model_path)
    try:
        # Only pass max_model_len for known model types
        if model_key in ["llama2", "mistral", "llama3"] and "max_model_len" in config:
            llm = LLM(model=model_path,
                      tokenizer_mode="auto",
                      trust_remote_code=True,
                      max_model_len=config["max_model_len"],
                      gpu_memory_utilization=0.95)
        elif model_key in ["qwen3"] and "max_model_len" in config:
            # Load Qwen3 model
            llm = LLM(model=model_path,
                      tokenizer_mode="auto",
                      trust_remote_code=True,
                      max_model_len=config["max_model_len"],
                      gpu_memory_utilization=0.95)
            # After loading, override tokenizer.chat_template
            import os
            template_path = os.path.join(os.path.dirname(__file__), "qwen3_nonthinking.jinja")
            with open(template_path, "r") as f:
                chat_template_content = f.read()
            tokenizer = llm.get_tokenizer()
            tokenizer.chat_template = chat_template_content
            logger.info(f"Custom chat template loaded from {template_path}")
        else:
            # For unknown models or models that don't require max_model_len, use defaults
            llm = LLM(model=model_path,
                      tokenizer_mode="auto",
                      trust_remote_code=True,
                      gpu_memory_utilization=0.95)

        # For non-Qwen3, get tokenizer here
        if model_key not in ["qwen3"]:
            tokenizer = llm.get_tokenizer()
        logger.info(f"Model {model_path} loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Start the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)