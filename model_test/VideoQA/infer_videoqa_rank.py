#!/usr/bin/env python3
"""One-process-per-GPU inference for the official 2026 ARTS VideoQA split."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers.video_utils import VideoMetadata


SYSTEM_PROMPT = (
    "Answer the multiple-choice question using the complete sampled UAV video. "
    "Return exactly one uppercase option letter from the choices shown in the question. "
    "Do not provide reasoning or any other text."
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    required = {"Question_id", "video_id", "question_category", "question", "answer"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Unexpected parquet schema in {path}")
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed[int(row["input_index"])] = row
            except Exception as exc:
                raise RuntimeError(f"Invalid resume row at {path}:{line_number}: {exc}") from exc
    return completed


def allowed_choices(question: str) -> list[str]:
    choices: list[str] = []
    for match in re.finditer(r"(?:^|\s)([A-G])\s*[\.\)]\s+", question.upper()):
        letter = match.group(1)
        if letter not in choices:
            choices.append(letter)
    if len(choices) < 2:
        raise ValueError(f"Could not identify choices in question: {question[:200]}")
    return choices


def extract_choice(output: str, allowed: list[str]) -> str | None:
    text = str(output).strip().upper()
    if "</THINK>" in text:
        text = text.rsplit("</THINK>", 1)[-1].strip()
    allowed_class = "".join(re.escape(letter) for letter in allowed)
    patterns = (
        rf"^\s*([{allowed_class}])\s*[\.]?\s*$",
        rf"(?:OPTION|ANSWER|CHOICE)\s*(?:IS|:)?\s*([{allowed_class}])\b",
        rf"\(([{allowed_class}])\)",
        rf"\b([{allowed_class}])\s*[\.]?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1)
    return None


def find_video(video_root: Path, video_id: str) -> Path:
    candidate = video_root / str(video_id)
    if candidate.is_file():
        return candidate
    if not candidate.suffix:
        for suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            with_suffix = Path(str(candidate) + suffix)
            if with_suffix.is_file():
                return with_suffix
    raise FileNotFoundError(f"Video not found for video_id={video_id}")


def sample_video(video_path: Path, max_frames: int) -> tuple[np.ndarray, VideoMetadata]:
    reader = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames <= 0:
        raise RuntimeError(f"No frames in {video_path}")
    sample_count = min(max_frames, total_frames)
    indices = np.unique(
        np.linspace(0, total_frames - 1, num=sample_count, dtype=np.int64)
    )
    frames = reader.get_batch(indices.tolist()).asnumpy()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"Unexpected decoded frame shape {frames.shape} for {video_path}")
    fps = float(reader.get_avg_fps())
    if not np.isfinite(fps) or fps <= 0:
        fps = 24.0
    metadata = VideoMetadata(
        total_num_frames=total_frames,
        fps=fps,
        width=int(frames.shape[2]),
        height=int(frames.shape[1]),
        duration=float(total_frames / fps),
        video_backend="decord",
        frames_indices=indices.tolist(),
    )
    return frames, metadata


def load_model(model_path: Path, device: torch.device):
    from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, processor, tokenizer


def infer_one(
    model,
    processor,
    tokenizer,
    video_path: Path,
    question: str,
    allowed: list[str],
    device: torch.device,
    max_frames: int,
    max_total_video_pixels: int,
    max_new_tokens: int,
) -> tuple[str, int]:
    frames, metadata = sample_video(video_path, max_frames=max_frames)
    prompt = (
        f"{SYSTEM_PROMPT}\nAllowed option letters: {', '.join(allowed)}.\n\n"
        f"{question.strip()}\n\nAnswer:"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = processor(
        text=[text],
        videos=[frames],
        videos_kwargs={
            "do_sample_frames": False,
            "video_metadata": [metadata],
            "size": {
                "shortest_edge": 128 * 32 * 32,
                "longest_edge": max_total_video_pixels,
            },
            "return_metadata": True,
        },
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    decoder = processor if hasattr(processor, "batch_decode") else tokenizer
    output = decoder.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return output, int(frames.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-parquet", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-total-video-pixels", type=int, default=6 * 1024 * 1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--save-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"Need {world_size} visible CUDA devices, found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    rows = load_rows(args.data_parquet)
    if args.limit > 0:
        rows = rows[: args.limit]
    assigned = [(index, item) for index, item in enumerate(rows) if index % world_size == rank]
    rank_path = args.result_dir / "ranks" / f"rank_{rank:02d}.jsonl"
    completed = load_completed(rank_path)
    model, processor, tokenizer = load_model(args.model_path, device)

    for input_index, item in tqdm(assigned, desc=f"rank {rank}", position=rank, leave=True):
        if input_index in completed:
            continue
        question_id = str(item["Question_id"])
        question = str(item["question"])
        allowed = allowed_choices(question)
        video_path = find_video(args.video_root, str(item["video_id"]))
        record: dict[str, Any] = {
            "input_index": input_index,
            "question_id": question_id,
            "video_id": str(item["video_id"]),
            "video_path": str(video_path),
            "question_category": str(item["question_category"]),
            "allowed_choices": allowed,
            "raw_output": "",
            "prediction": None,
            "correct_answer": str(item["answer"]).upper(),
            "correct": False,
            "sampled_frames": 0,
            "model_path": str(args.model_path),
            "rank": rank,
            "error": None,
        }
        try:
            raw_output, sampled_frames = infer_one(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                video_path=video_path,
                question=question,
                allowed=allowed,
                device=device,
                max_frames=args.max_frames,
                max_total_video_pixels=args.max_total_video_pixels,
                max_new_tokens=args.max_new_tokens,
            )
            prediction = extract_choice(raw_output, allowed)
            record["raw_output"] = raw_output
            record["prediction"] = prediction
            record["sampled_frames"] = sampled_frames
            record["correct"] = prediction == record["correct_answer"]
            if prediction is None:
                record["error"] = "Could not parse a legal option letter from model output"
        except Exception as exc:
            record["error"] = repr(exc)
            print(
                f"rank={rank} index={input_index} question_id={question_id} error={exc!r}",
                flush=True,
            )
            if not args.save_errors:
                raise
        append_jsonl(rank_path, record)

    print(f"rank={rank} assigned={len(assigned)} output={rank_path}", flush=True)


if __name__ == "__main__":
    main()
