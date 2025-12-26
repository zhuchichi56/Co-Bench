#!/usr/bin/env python3
"""
MT-Bench Standalone Judge

A standalone implementation of MT-Bench evaluation system that consolidates all
necessary components into a single file without external imports from fastchat.
"""

import os
import json
import time
import ast
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import fire

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"
TIE_DELTA = 0.1

# Model lists
OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "chatgpt-4o-latest-20240903",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-mini",
)

ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-instant-1",
    "claude-instant-1.2",
)

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

# Score extraction patterns
two_score_pattern = re.compile(r"\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile(r"\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

# Temperature configuration
temperature_config = {
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

# TODO：Built-in judge prompts 请检查这个judge propmt 是不是完全一样的
JUDGE_PROMPTS = {
    "single-v1": {
        "name": "single-v1",
        "type": "single",
        "system_prompt": "You are a helpful assistant.",
        "prompt_template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]",
        "description": "Prompt for general questions",
        "category": "general",
        "output_format": "[[rating]]"
    },
    "single-math-v1": {
        "name": "single-math-v1",
        "type": "single",
        "system_prompt": "You are a helpful assistant.",
        "prompt_template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]",
        "description": "Prompt for math questions",
        "category": "math",
        "output_format": "[[rating]]"
    },
    "single-v1-multi-turn": {
        "name": "single-v1-multi-turn",
        "type": "single",
        "system_prompt": "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant's answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n",
        "prompt_template": "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>",
        "description": "Prompt for general questions",
        "category": "general",
        "output_format": "[[rating]]"
    },
    "single-math-v1-multi-turn": {
        "name": "single-math-v1-multi-turn",
        "type": "single",
        "system_prompt": "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n",
        "prompt_template": "<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>",
        "description": "Prompt for math questions",
        "category": "math",
        "output_format": "[[rating]]"
    }
}

# Default fallback questions (will try to load from original path)
DEFAULT_MT_BENCH_QUESTIONS = [
    {"question_id": 81, "category": "writing", "turns": ["Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.", "Rewrite your previous response. Start every sentence with the letter A."]},
    {"question_id": 82, "category": "writing", "turns": ["Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.", "Take a moment to evaluate and critique your own response."]}
]

@dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False

@dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False

@dataclass
class Conversation:
    """Simple conversation template for API calls"""
    name: str
    system_message: str = ""
    messages: List[List[str]] = None
    roles: Tuple[str] = ("USER", "ASSISTANT")

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def set_system_message(self, message: str):
        self.system_message = message

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def to_openai_api_messages(self):
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})

        for role, content in self.messages:
            if role == self.roles[0]:  # USER
                api_role = "user"
            elif role == self.roles[1]:  # ASSISTANT
                api_role = "assistant"
            else:
                api_role = role.lower()

            if content is not None:
                messages.append({"role": api_role, "content": content})

        return messages

def get_conversation_template(model: str) -> Conversation:
    """Get conversation template for model"""
    return Conversation(name=model, roles=("USER", "ASSISTANT"))

def chat_completion_openai(model: str, conv: Conversation, temperature: float, max_tokens: int, api_dict: dict = None):
    """Make OpenAI API call"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")

    if api_dict is not None:
        client = OpenAI(api_key=api_dict["api_key"], base_url=api_dict["api_base"])
    else:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )

    output = API_ERROR_OUTPUT

    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except Exception as e:
            logger.error(f"OpenAI API error: {type(e).__name__}: {e}")
            time.sleep(API_RETRY_SLEEP)

    return output

def chat_completion_anthropic(model: str, conv: Conversation, temperature: float, max_tokens: int, api_dict: dict = None):
    """Make Anthropic API call"""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            prompt = conv.get_prompt()
            response = c.completions.create(
                model=model,
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            output = response.completion
            break
        except Exception as e:
            logger.error(f"Anthropic API error: {type(e).__name__}: {e}")
            time.sleep(API_RETRY_SLEEP)
    return output.strip()

def run_judge_single(question: dict, answer: dict, judge: Judge, ref_answer: dict = None, multi_turn: bool = False):
    """Run single judge evaluation"""
    kwargs = {}
    model = judge.model_name

    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = get_conversation_template(model)
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if model in OPENAI_MODEL_LIST:
        judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048)
    elif model in ANTHROPIC_MODEL_LIST:
        judgment = chat_completion_anthropic(model, conv, temperature=0, max_tokens=1024)
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[rating]]":
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(f"Invalid output format: {judge.prompt_template['output_format']}")

    return rating, user_prompt, judgment

def play_a_match_single(match: MatchSingle, output_file: str = None):
    """Play a single match"""
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "single":
        score, user_prompt, judgment = run_judge_single(
            question, answer, judge, ref_answer, multi_turn=multi_turn
        )

        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2
        result = {
            "question_id": question_id,
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": turn,
            "tstamp": time.time(),
        }
        logger.info(f"question: {question_id}, turn: {turn}, model: {model}, score: {score}")
    else:
        raise ValueError(f"Invalid judge type: {judge.prompt_template['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result) + "\n")

    return result

def load_questions_from_file(question_file: str = None):
    """Load questions from JSONL file"""
    questions = []

    # Try to load from specified file first
    if question_file:
        try:
            with open(question_file, "r") as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            return questions
        except FileNotFoundError:
            logger.warning(f"Question file {question_file} not found")

    # Try to load from original fastchat path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_question_file = os.path.join(script_dir, "fastchat", "llm_judge", "data", "mt_bench", "question.jsonl")

    try:
        with open(original_question_file, "r") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        logger.info(f"Loaded {len(questions)} questions from {original_question_file}")
        return questions
    except FileNotFoundError:
        logger.warning(f"Original question file {original_question_file} not found, using default questions")
        return DEFAULT_MT_BENCH_QUESTIONS

def load_model_answers_from_file(answer_file: str):
    """Load model answers from JSONL file"""
    model_answers = {}

    try:
        with open(answer_file, "r") as f:
            for line in f:
                if line.strip():
                    answer = json.loads(line)
                    model_name = answer.get("model_id", "unknown")
                    if model_name not in model_answers:
                        model_answers[model_name] = {}
                    model_answers[model_name][answer["question_id"]] = answer
    except FileNotFoundError:
        raise FileNotFoundError(f"Answer file {answer_file} not found")

    return model_answers

def make_judge_single(judge_model: str):
    """Create single judges"""
    judges = {}
    judges["default"] = Judge(judge_model, JUDGE_PROMPTS["single-v1"])
    judges["math"] = Judge(judge_model, JUDGE_PROMPTS["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(judge_model, JUDGE_PROMPTS["single-v1-multi-turn"], multi_turn=True)
    judges["math-mt"] = Judge(judge_model, JUDGE_PROMPTS["single-math-v1-multi-turn"], ref_based=True, multi_turn=True)
    return judges

def make_match_single(questions: List[dict], models: List[str], model_answers: Dict, judge: Judge, ref_answers: Dict = None, multi_turn: bool = False):
    """Create single matches"""
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for model in models:
            if model not in model_answers:
                logger.warning(f"No answers found for model: {model}")
                continue
            q_id = q["question_id"]
            if q_id not in model_answers[model]:
                logger.warning(f"No answer found for model {model}, question {q_id}")
                continue
            answer = model_answers[model][q_id]
            ref_answer = None
            if ref_answers and judge.model_name in ref_answers and q_id in ref_answers[judge.model_name]:
                ref_answer = ref_answers[judge.model_name][q_id]

            match = MatchSingle(dict(q), model, answer, judge, ref_answer, multi_turn)
            matches.append(match)
    return matches

def evaluate_responses(
    input_file: str,
    output_file: str = None,
    judge_model: str = "gpt-4o-mini-2024-07-18",
    question_file: str = None,
    parallel: int = 1,
    first_n: int = None
):
    """
    Evaluate responses using MT-Bench methodology

    Args:
        input_file: Path to JSONL file containing instruction and response pairs
        output_file: Path to output JSONL file for results
        judge_model: Model to use for judging
        question_file: Path to questions file (optional, uses built-in if not provided)
        parallel: Number of parallel API calls
        first_n: Only evaluate first n questions (for debugging)
    """

    # Load questions
    questions = load_questions_from_file(question_file)

    if first_n:
        questions = questions[:first_n]

    # Load model answers
    model_answers = load_model_answers_from_file(input_file)
    models = list(model_answers.keys())

    logger.info(f"Loaded {len(questions)} questions")
    logger.info(f"Found {len(models)} models: {models}")

    # Create judges
    judges = make_judge_single(judge_model)

    # Load reference answers (dummy implementation - you may want to load from file)
    ref_answers = None  # You can implement this if you have reference answers

    # Create matches
    matches = []
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]
    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]

    matches += make_match_single(question_default, models, model_answers, judges["default"])
    matches += make_match_single(question_math, models, model_answers, judges["math"], ref_answers)
    matches += make_match_single(question_default, models, model_answers, judges["default-mt"], multi_turn=True)
    matches += make_match_single(question_math, models, model_answers, judges["math-mt"], ref_answers, multi_turn=True)

    logger.info(f"Created {len(matches)} evaluation matches")

    # Set default output file if not provided
    if output_file is None:
        output_file = f"mt_bench_judgment_{judge_model.replace('/', '_')}_single.jsonl"

    # Run evaluation
    if parallel == 1:
        for match in tqdm(matches, desc="Evaluating"):
            play_a_match_single(match, output_file=output_file)
    else:
        def play_match_wrapper(match):
            return play_a_match_single(match, output_file=output_file)

        with ThreadPoolExecutor(parallel) as executor:
            list(tqdm(executor.map(play_match_wrapper, matches), total=len(matches), desc="Evaluating"))

    logger.info(f"Evaluation completed. Results saved to {output_file}")
    return output_file

def show_results(input_file: str, model_list: List[str] = None):
    """Show evaluation results"""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed. Install with: pip install pandas")
        return

    with open(input_file, "r") as f:
        judge_data = [json.loads(line) for line in f if line.strip()]

    df_all = pd.DataFrame(judge_data)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if model_list:
        df = df[df["model"].isin(model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    print("\n########## Second turn ##########")
    df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
    if not df_2.empty:
        print(df_2.sort_values(by="score", ascending=False))

    print("\n########## Average ##########")
    df_3 = df[["model", "score"]].groupby(["model"]).mean()
    print(df_3.sort_values(by="score", ascending=False))

# Fire CLI interface
def main():
    """Main CLI interface using Fire"""
    return fire.Fire({
        'evaluate': evaluate_responses,
        'show': show_results
    })

if __name__ == "__main__":
    main()