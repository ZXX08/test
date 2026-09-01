import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFAULT_RESULT_PATH = "only_o3dvqa/only_o3dvqa_responses.json"
DEFAULT_DATA_JSON = "/root/workspace-sj/O3DVQA/Test.json"
DEFAULT_JUDGE_MODEL_NAME = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://www.autodl.art/api/v1"
NUMERIC_LOWER_RATIO = 0.75
NUMERIC_UPPER_RATIO = 1.25


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_progress(log_file: str | Path, index: int | str, correct: int, total: int):
    ensure_dir(Path(log_file).parent)
    accuracy = correct / total if total > 0 else 0.0
    with open(log_file, "a", encoding="utf-8") as logf:
        logf.write(f"[{datetime.now()}] Step {index}: Accuracy = {accuracy:.4f} ({correct}/{total})\n")


def build_judge_prompt(question: str, answer: str, response: str) -> str:
    return f"""You should help me evaluate whether the model response matches the correct answer.
Output exactly one integer: 1 or 0.
1 means the response matches the answer.
0 means the response is wrong or conflicts with the answer.
Do not output explanations.

Question: {question}
Correct answer: {answer}
Model response: {response}
Score:"""


def build_quantitative_judge_prompt(question: str, answer: str, response: str) -> str:
    return f"""You should evaluate an absolute measurement question, such as object distance or size estimation.
First, extract the numeric value from the correct answer.
Second, extract the numeric value from the model response.
Then compare the extracted response value with the extracted correct value.
If response_value / correct_value is in the range [{NUMERIC_LOWER_RATIO}, {NUMERIC_UPPER_RATIO}], output 1.
Otherwise, output 0.
If either value cannot be extracted, output 0.
Output exactly one integer: 1 or 0.
Do not output explanations, extracted values, or calculations.

Question: {question}
Correct answer: {answer}
Model response: {response}
Score:"""


def parse_score(text: str) -> str | None:
    text = str(text).strip()
    match = re.search(r"\b([01])\b", text)
    if match:
        return match.group(1)
    if text.startswith("1"):
        return "1"
    if text.startswith("0"):
        return "0"
    return None


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\"']+|[\s\"'.]+$", "", text)
    return text


def get_question_type(item: dict[str, Any], question_type_by_id: dict[str, str] | None = None) -> str:
    question_type = item.get("question_type")
    if question_type:
        return str(question_type).strip().lower()
    if question_type_by_id and item.get("id") is not None:
        question_type = question_type_by_id.get(str(item["id"]))
        if question_type:
            return str(question_type).strip().lower()
    qa_info = item.get("qa_info")
    if isinstance(qa_info, dict):
        return str(qa_info.get("type", "")).strip().lower()
    return ""


def check_match(
    client: OpenAI | None,
    judge_model_name: str,
    question: str,
    answer: str,
    response: str,
    question_type: str,
    timeout: int,
) -> tuple[str, str]:
    if normalize_text(answer) == normalize_text(response):
        return "1", "exact_match"

    if client is None:
        raise RuntimeError("AutoDL API key is required for GPT-based judging.")

    prompt = (
        build_quantitative_judge_prompt(question, answer, response)
        if question_type == "quantitative"
        else build_judge_prompt(question, answer, response)
    )

    raw_score = client.chat.completions.create(
        timeout=timeout,
        model=judge_model_name,
        messages=[
            {"role": "system", "content": "You are a strict QA evaluator. Output only 0 or 1."},
            {"role": "user", "content": prompt},
        ],
    ).choices[0].message.content

    score = parse_score(raw_score)
    if score is None:
        raise ValueError(f"Invalid judge score: {raw_score!r}")
    return score, raw_score


def flatten_results(data: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    all_items = []
    for dataset_name, scenes in data.items():
        for scene_name, items in scenes.items():
            for item in items:
                all_items.append((dataset_name, scene_name, item))
    return all_items


def load_question_types(data_json: str | Path | None) -> dict[str, str]:
    if not data_json or not Path(data_json).exists():
        return {}
    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data if isinstance(data, list) else []
    return {
        str(item["id"]): str(item.get("question_type", "")).strip().lower()
        for item in rows
        if isinstance(item, dict) and item.get("id") is not None
    }


def evaluate(args: argparse.Namespace):
    with open(args.result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log_file = Path(args.log_dir) / f"{args.judge_model_name}.txt"
    detail_path = Path(args.log_dir) / f"{args.judge_model_name}_details.jsonl"
    ensure_dir(Path(args.log_dir))
    if not args.resume:
        for path in [log_file, detail_path]:
            if path.exists():
                path.unlink()

    api_key = args.api_key or os.getenv("AUTODL_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=args.base_url) if api_key else None
    all_items = flatten_results(data)
    question_type_by_id = load_question_types(args.data_json)

    total = 0
    correct = 0
    for i, (dataset, scene, item) in enumerate(tqdm(all_items, desc="Evaluating"), 1):
        if args.limit > 0 and total >= args.limit:
            break

        question = item.get("question", "")
        pred = item.get("response", "")
        gt = item.get("answer", "")
        question_type = get_question_type(item, question_type_by_id)
        if not pred or not gt:
            continue

        try:
            score, raw_score = check_match(
                client=client,
                judge_model_name=args.judge_model_name,
                question=question,
                answer=gt,
                response=pred,
                question_type=question_type,
                timeout=args.timeout,
            )
        except Exception as exc:
            score = "0"
            raw_score = repr(exc)

        correct += 1 if score == "1" else 0
        total += 1
        log_progress(log_file, i, correct, total)
        with open(detail_path, "a", encoding="utf-8") as detail_f:
            detail_f.write(
                json.dumps(
                    {
                        "index": i,
                        "dataset": dataset,
                        "scene": scene,
                        "id": item.get("id"),
                        "question": question,
                        "question_type": question_type,
                        "answer": gt,
                        "response": pred,
                        "score": score,
                        "raw_judge_output": raw_score,
                        "judge_model": args.judge_model_name,
                        "base_url": args.base_url,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
    log_progress(log_file, "Final", correct, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Open3D-VQA responses with AutoDL gpt-5.4-mini.")
    parser.add_argument("--result-path", default=DEFAULT_RESULT_PATH)
    parser.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    parser.add_argument("--judge-model-name", default=DEFAULT_JUDGE_MODEL_NAME)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--log-dir", default="only_o3dvqa")
    parser.add_argument("--timeout", type=int, default=75)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())


