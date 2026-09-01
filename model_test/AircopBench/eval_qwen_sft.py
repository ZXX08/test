#!/usr/bin/env python3
"""Run local Qwen SFT evaluation on AirCopBench.

This is a standalone, non-invasive runner. It mirrors the data loading, prompt
construction, multi-image input, option extraction, and accuracy aggregation in:

- code/AirCopBench_evaluation/run_qwen25vl.py
- code/AirCopBench_evaluation/eval_qwen25vl.py

No existing benchmark files are modified.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig


DEFAULT_MODEL_PATH = ""
DEFAULT_DATA_ROOT = ""
DEFAULT_OUTPUT_DIR = ""

DATASET_FILES = {
    "Real2_VQA": "Real2_VQA_test.json",
    "Sim3_VQA": "Sim3_VQA_test.json",
    "Sim5_VQA": "Sim5_VQA_test.json",
    "Sim6_VQA": "Sim6_VQA_test.json",
}

CATEGORY_MAP = {
    "Scene Understanding": [
        "Scene Description",
        "Scene Comparison",
        "Observing Posture",
    ],
    "Object Understanding": [
        "Object Recognition",
        "Object Counting",
        "Object Grounding",
        "Object Matching",
    ],
    "Perception Assessment": [
        "Quality Assessment",
        "Usability Assessment",
        "Causal Assessment",
    ],
    "Collaborative Decision": [
        "When to Collaborate",
        "What to Collaborate",
        "Who to Collaborate",
        "Why to Collaborate",
    ],
}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, data: Any) -> None:
    ensure_dir(Path(path).parent)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str | Path, item: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_done_ids(path: str | Path) -> set[str]:
    return {str(x.get("sample_id")) for x in load_jsonl(path) if x.get("sample_id") is not None}


def parse_subcat(qtype: Any) -> str:
    if not isinstance(qtype, str):
        return str(qtype)
    match = re.match(r"\d+\.\d+\s+(.+?)\s*(?:\(|$)", qtype)
    return match.group(1).strip() if match else qtype.strip()


def category_for_subcategory(subcategory: str) -> str | None:
    for category, subs in CATEGORY_MAP.items():
        if subcategory in subs:
            return category
    return None


def extract_answer(response: str) -> str | None:
    txt = str(response).strip().upper()
    if "</THINK>" in txt:
        txt = txt.split("</THINK>")[-1].strip()
    patterns = [
        r"ANSWER:\s*([ABCD])",
        r"THE ANSWER IS\s*([ABCD])",
        r"\(([ABCD])\)",
        r"^([ABCD]):",
        r"OPTION\s*([ABCD])",
        r"([ABCD])\s*\.?\s*$",
        r"\b([ABCD])\b",
    ]
    for pat in patterns:
        match = re.search(pat, txt, re.MULTILINE)
        if match:
            return match.group(1)
    return None


def build_prompt(question: str, options: dict[str, Any]) -> str:
    opts = "\n".join([f"{k}. {v}" for k, v in options.items()])
    return f"{question}\n{opts}\nAnswer with the option's letter from the given choices directly."


def get_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str):
    print(f"Loading model from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"config.model_type: {getattr(config, 'model_type', None)}")
    print(f"config.architectures: {getattr(config, 'architectures', None)}")
    print(f"processor class: {processor.__class__.__name__}")
    print(f"tokenizer class: {tokenizer.__class__.__name__}")

    auto_model_candidates = []
    for cls_name in ["AutoModelForImageTextToText", "AutoModelForVision2Seq"]:
        try:
            module = __import__("transformers", fromlist=[cls_name])
            auto_model_candidates.append(getattr(module, cls_name))
        except Exception:
            continue
    auto_model_candidates.append(AutoModel)

    last_error = None
    for model_cls in auto_model_candidates:
        try:
            model = model_cls.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            try:
                model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                pass
            model.eval()
            print(f"model class: {model.__class__.__name__}")
            print(f"model has generate: {hasattr(model, 'generate')}")
            if model_cls is AutoModel and not hasattr(model, "generate"):
                raise RuntimeError(
                    "Loaded AutoModel successfully, but this class has no generate(); please use a transformers "
                    "version that supports Qwen3_5ForConditionalGeneration or AutoModelForImageTextToText."
                )
            return model, processor, tokenizer
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load model from {model_path}: {last_error}")


def infer_one(
    model,
    processor,
    tokenizer,
    image_paths: list[str],
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        images.append(Image.open(image_path).convert("RGB"))
    if not images:
        raise FileNotFoundError("No valid images found for this sample.")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    device = get_device(model)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    input_ids_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_ids_len:]
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def iter_aircop_samples(data_root: str):
    for dataset, rel_path in DATASET_FILES.items():
        path = Path(data_root) / rel_path
        if not path.exists():
            print(f"Dataset file not found, skip: {path}")
            continue
        data = read_json(path)
        if isinstance(data, dict):
            items = data.get("results")
            if items is None:
                print(f"JSON dict has no results field, skip: {path}")
                continue
        elif isinstance(data, list):
            items = data
        else:
            print(f"Unexpected JSON type {type(data)}, skip: {path}")
            continue
        for item in items:
            if isinstance(item, dict):
                yield dataset, item


def normalize_sample(item: dict[str, Any], data_root: str, fallback_id: int) -> dict[str, Any]:
    """Normalize both the original benchmark schema and LlamaFactory ShareGPT schema."""
    if isinstance(item.get("conversations"), list):
        conversations = item["conversations"]
        human = next((turn.get("value", "") for turn in conversations if turn.get("from") == "human"), "")
        assistant = next((turn.get("value", "") for turn in conversations if turn.get("from") == "gpt"), "")
        prompt = re.sub(r"(?:<image>\s*)+", "", str(human)).strip()
        image_paths = [str(Path(data_root) / path) for path in item.get("image", [])]
        return {
            "id": item.get("id", fallback_id),
            "question": prompt,
            "prompt": f"{prompt}\nAnswer with the option's letter from the given choices directly.",
            "image_paths": image_paths,
            "ground_truth": extract_answer(str(assistant)),
        }

    options = item.get("options", {})
    uav_paths = item.get("uav_paths", {})
    image_paths = [str(Path(data_root) / path) for path in uav_paths.values()] if isinstance(uav_paths, dict) else []
    question = item.get("question", "")
    return {
        "id": item.get("question_id", fallback_id),
        "question": question,
        "prompt": build_prompt(question, options if isinstance(options, dict) else {}),
        "image_paths": image_paths,
        "ground_truth": item.get("correct_answer"),
    }


def compute_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if r.get("parsed_answer") == r.get("ground_truth"))
    return correct / len(rows)


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if r.get("error") is None and r.get("ground_truth") is not None]
    datasets_done = sorted({r.get("dataset") for r in valid if r.get("dataset")})

    per_dataset = {}
    per_subcategory_rows: dict[str, list[dict[str, Any]]] = {}
    per_dataset_subcategory_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for dataset in datasets_done:
        ds_rows = [r for r in valid if r.get("dataset") == dataset]
        per_dataset[dataset] = compute_accuracy(ds_rows)
        for row in ds_rows:
            sub = row.get("subcategory") or parse_subcat(row.get("question_type", ""))
            per_subcategory_rows.setdefault(sub, []).append(row)
            per_dataset_subcategory_rows.setdefault((dataset, sub), []).append(row)

    per_subcategory = {sub: compute_accuracy(rows) for sub, rows in per_subcategory_rows.items()}
    per_dataset_subcategory = {
        f"{dataset}----{sub}": compute_accuracy(rows)
        for (dataset, sub), rows in per_dataset_subcategory_rows.items()
    }

    per_category = {}
    per_dataset_category = {}
    for category, subs in CATEGORY_MAP.items():
        cat_rows = [r for r in valid if (r.get("subcategory") or parse_subcat(r.get("question_type", ""))) in subs]
        per_category[category] = compute_accuracy(cat_rows)
        for dataset in datasets_done:
            ds_cat_rows = [
                r
                for r in valid
                if r.get("dataset") == dataset
                and (r.get("subcategory") or parse_subcat(r.get("question_type", ""))) in subs
            ]
            per_dataset_category[f"{dataset}----{category}"] = compute_accuracy(ds_cat_rows)

    return {
        "overall_accuracy": compute_accuracy(valid),
        "total_records": len(records),
        "valid_records": len(valid),
        "error_records": len([r for r in records if r.get("error") is not None]),
        "per_dataset": per_dataset,
        "per_subcategory": per_subcategory,
        "per_category": per_category,
        "per_dataset_subcategory": per_dataset_subcategory,
        "per_dataset_category": per_dataset_category,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local Qwen SFT on AirCopBench.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    jsonl_path = output_dir / "results.jsonl"
    results_json_path = output_dir / "results.json"
    scores_json_path = output_dir / "scores.json"
    log_path = output_dir / "run.log"

    generation_config = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }

    done = load_done_ids(jsonl_path) if args.resume else set()
    model, processor, tokenizer = load_model(args.model_path)

    processed = 0
    for dataset, item in tqdm(list(iter_aircop_samples(args.data_root)), desc="AirCopBench"):
        if args.limit > 0 and processed >= args.limit:
            break

        sample = normalize_sample(item, args.data_root, processed)
        qid = sample["id"]
        sample_id = f"{dataset}:{qid}"
        if sample_id in done:
            continue

        question_type = item.get("question_type", "")
        subcategory = parse_subcat(question_type)
        category = category_for_subcategory(subcategory)
        options = item.get("options", {})
        image_paths = sample["image_paths"]
        prompt = sample["prompt"]

        record = {
            "benchmark": "AirCopBench",
            "sample_id": sample_id,
            "dataset": dataset,
            "category": category,
            "subcategory": subcategory,
            "question": sample["question"],
            "options": options,
            "image_paths": image_paths,
            "raw_output": None,
            "parsed_answer": None,
            "ground_truth": sample["ground_truth"],
            "correct": None,
            "error": None,
            "model_path": args.model_path,
            "generation_config": generation_config,
            "question_type": question_type,
        }

        try:
            raw_output = infer_one(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            parsed_answer = extract_answer(raw_output)
            record["raw_output"] = raw_output
            record["parsed_answer"] = parsed_answer
            record["correct"] = parsed_answer == sample["ground_truth"] if parsed_answer is not None else False
        except Exception as exc:
            record["error"] = repr(exc)
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(f"{sample_id}\t{repr(exc)}\n")

        append_jsonl(jsonl_path, record)
        processed += 1

    records = load_jsonl(jsonl_path)
    dump_json(results_json_path, records)
    dump_json(scores_json_path, aggregate_metrics(records))
    print(f"Results saved to: {results_json_path}")
    print(f"Scores saved to: {scores_json_path}")


if __name__ == "__main__":
    main()


