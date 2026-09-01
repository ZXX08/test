#!/usr/bin/env python3
"""Evaluate a local Qwen3.5 vision-language model on AirCopBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from eval_qwen_sft import (
    aggregate_metrics,
    append_jsonl,
    category_for_subcategory,
    dump_json,
    extract_answer,
    infer_one,
    load_done_ids,
    load_jsonl,
    load_model,
    parse_subcat,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS_DIR = SCRIPT_DIR
DEFAULT_IMAGES_ROOT = SCRIPT_DIR / "AircopBench"
DEFAULT_MODEL_PATH = Path("/root/workspace-sj/saves/qwen3vl_stage2_urbanvideobench")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "qwen3vl_stage2"

DATASET_FILES = {
    "Real2_VQA": "Real2_VQA_test.json",
    "Sim3_VQA": "Sim3_VQA_test.json",
    "Sim5_VQA": "Sim5_VQA_test.json",
    "Sim6_VQA": "Sim6_VQA_test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local Qwen3.5 SFT model on AirCopBench")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=-1, help="Evaluate only the first N pending samples")
    parser.add_argument("--resume", action="store_true", help="Continue from an existing results.jsonl")
    parser.add_argument("--check-only", action="store_true", help="Validate data without loading the model")
    return parser.parse_args()


def load_samples(annotations_dir: Path, images_root: Path) -> list[dict]:
    samples = []
    for dataset, filename in DATASET_FILES.items():
        annotation_path = annotations_dir / filename
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError(f"Expected a JSON list in {annotation_path}, got {type(rows).__name__}")

        for index, item in enumerate(rows):
            conversations = item.get("conversations", [])
            human = next((x.get("value", "") for x in conversations if x.get("from") == "human"), "")
            assistant = next((x.get("value", "") for x in conversations if x.get("from") == "gpt"), "")
            question = human.replace("<image>\n", "").replace("<image>", "").strip()
            image_paths = [images_root / relative_path for relative_path in item.get("image", [])]
            samples.append(
                {
                    "dataset": dataset,
                    "sample_id": f"{dataset}:{item.get('id', index)}",
                    "question_type": item.get("question_type", ""),
                    "question": question,
                    "prompt": f"{question}\nAnswer with the option's letter from the given choices directly.",
                    "image_paths": image_paths,
                    "ground_truth": extract_answer(str(assistant)),
                }
            )
    return samples


def validate_inputs(args: argparse.Namespace, samples: list[dict]) -> None:
    if not args.model_path.is_dir() and not args.check_only:
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if args.limit == 0 or args.limit < -1:
        raise ValueError("--limit must be -1 or a positive integer")

    missing_images = sorted(
        {str(path) for sample in samples for path in sample["image_paths"] if not path.is_file()}
    )
    invalid_labels = [sample["sample_id"] for sample in samples if sample["ground_truth"] not in {"A", "B", "C", "D"}]
    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"Missing {len(missing_images)} referenced images. First entries:\n{preview}")
    if invalid_labels:
        raise ValueError(f"Invalid answer labels in {len(invalid_labels)} samples: {invalid_labels[:10]}")

    image_count = sum(len(sample["image_paths"]) for sample in samples)
    print(f"Data check passed: {len(samples)} samples, {image_count} image references")
    print(f"Annotations: {args.annotations_dir}")
    print(f"Images: {args.images_root}")


def main() -> None:
    args = parse_args()
    samples = load_samples(args.annotations_dir, args.images_root)
    validate_inputs(args, samples)
    if args.check_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "results.jsonl"
    done_ids = load_done_ids(jsonl_path) if args.resume else set()
    pending = [sample for sample in samples if sample["sample_id"] not in done_ids]
    if args.limit > 0:
        pending = pending[: args.limit]
    if not pending:
        print("No pending samples.")
        return

    model, processor, tokenizer = load_model(str(args.model_path))
    for sample in tqdm(pending, desc="AirCopBench"):
        subcategory = parse_subcat(sample["question_type"])
        record = {
            "benchmark": "AirCopBench",
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "category": category_for_subcategory(subcategory),
            "subcategory": subcategory,
            "question": sample["question"],
            "image_paths": [str(path) for path in sample["image_paths"]],
            "raw_output": None,
            "parsed_answer": None,
            "ground_truth": sample["ground_truth"],
            "correct": False,
            "error": None,
            "model_path": str(args.model_path),
        }
        try:
            raw_output = infer_one(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image_paths=record["image_paths"],
                prompt=sample["prompt"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            record["raw_output"] = raw_output
            record["parsed_answer"] = extract_answer(raw_output)
            record["correct"] = record["parsed_answer"] == record["ground_truth"]
        except Exception as error:
            record["error"] = repr(error)
        append_jsonl(jsonl_path, record)

    records = load_jsonl(jsonl_path)
    dump_json(args.output_dir / "results.json", records)
    scores = aggregate_metrics(records)
    dump_json(args.output_dir / "scores.json", scores)
    print(f"Overall accuracy: {scores['overall_accuracy']:.4f}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()


