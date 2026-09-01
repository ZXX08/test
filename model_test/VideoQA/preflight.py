#!/usr/bin/env python3
"""Validate the official VideoQA manifest, videos, model, and GPU resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch


EXPECTED_PARQUET_SHA256 = "e28288b9c80824161fe2b2c846485a5901c4c4c463b5b9d3ffb6a0176f18f0c6"
EXPECTED_COUNT = 1071


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_video(video_root: Path, video_id: str) -> Path | None:
    candidate = video_root / video_id
    if candidate.is_file():
        return candidate
    if not candidate.suffix:
        for suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            path = Path(str(candidate) + suffix)
            if path.is_file():
                return path
    return None


def choices(question: str) -> set[str]:
    return set(re.findall(r"(?:^|\s)([A-G])\s*[\.\)]\s+", question.upper()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-parquet", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--min-free-mib", type=int, default=0)
    args = parser.parse_args()

    parquet_hash = sha256(args.data_parquet)
    if parquet_hash != EXPECTED_PARQUET_SHA256:
        raise RuntimeError(f"Official parquet SHA256 mismatch: {parquet_hash}")
    rows = pq.read_table(args.data_parquet).to_pylist()
    ids = [str(row["Question_id"]) for row in rows]
    duplicate_ids = [qid for qid, count in Counter(ids).items() if count > 1]
    if len(rows) != EXPECTED_COUNT or duplicate_ids:
        raise RuntimeError(f"Manifest count/ID failure: rows={len(rows)} duplicates={duplicate_ids[:10]}")

    invalid_answers = []
    missing_videos = []
    for row in rows:
        allowed = choices(str(row["question"]))
        answer = str(row["answer"]).upper()
        if answer not in allowed:
            invalid_answers.append(str(row["Question_id"]))
        if find_video(args.video_root, str(row["video_id"])) is None:
            missing_videos.append(str(row["video_id"]))
    if invalid_answers or missing_videos:
        raise RuntimeError(
            f"invalid_answers={invalid_answers[:10]} missing_videos={missing_videos[:10]}"
        )

    model_file = args.model_path / "model.safetensors"
    model_hash = sha256(model_file)
    for filename in (
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if not (args.model_path / filename).is_file():
            raise RuntimeError(f"Missing model file: {args.model_path / filename}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError(f"Expected exactly 8 visible GPUs, found {torch.cuda.device_count()}")
    free_mib = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free_bytes, _ = torch.cuda.mem_get_info()
        free_mib.append(int(free_bytes / 1024 / 1024))
    if args.min_free_mib and min(free_mib) < args.min_free_mib:
        raise RuntimeError(
            f"Insufficient free GPU memory: free_mib={free_mib}, required={args.min_free_mib}"
        )

    print(
        json.dumps(
            {
                "data_parquet": str(args.data_parquet),
                "parquet_sha256": parquet_hash,
                "questions": len(rows),
                "unique_question_ids": len(set(ids)),
                "unique_videos": len({str(row["video_id"]) for row in rows}),
                "missing_videos": 0,
                "answer_distribution": dict(sorted(Counter(str(row["answer"]) for row in rows).items())),
                "model_path": str(args.model_path),
                "model_sha256": model_hash,
                "cuda_devices": torch.cuda.device_count(),
                "gpu_free_mib": free_mib,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
