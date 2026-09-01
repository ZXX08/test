#!/usr/bin/env python3
"""Merge VideoQA ranks, calculate local accuracy, and create VideoQA.json."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def read_rank_rows(rank_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(rank_dir.glob("rank_*.jsonl"))
    if len(paths) != 8:
        raise RuntimeError(f"Expected 8 rank files, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-parquet", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = pq.read_table(args.data_parquet).to_pylist()
    expected_ids = [str(item["Question_id"]) for item in manifest]
    expected_by_id = {str(item["Question_id"]): item for item in manifest}
    rows = read_rank_rows(args.result_dir / "ranks")
    result_ids = [str(row.get("question_id", "")) for row in rows]

    problems = {
        "expected_count": len(expected_ids),
        "result_count": len(rows),
        "expected_unique": len(set(expected_ids)),
        "result_unique": len(set(result_ids)),
        "duplicate_results": [qid for qid, count in Counter(result_ids).items() if count > 1],
        "missing": sorted(set(expected_ids) - set(result_ids)),
        "extra": sorted(set(result_ids) - set(expected_ids)),
        "error_count": sum(bool(row.get("error")) for row in rows),
        "invalid_prediction_count": sum(
            row.get("prediction") not in set(row.get("allowed_choices", [])) for row in rows
        ),
    }
    if (
        problems["result_count"] != problems["expected_count"]
        or problems["duplicate_results"]
        or problems["missing"]
        or problems["extra"]
        or problems["error_count"]
        or problems["invalid_prediction_count"]
    ):
        write_json(args.result_dir / "validation_failed.json", problems)
        raise RuntimeError(f"VideoQA validation failed: {problems}")

    rows.sort(key=lambda row: int(row["input_index"]))
    correct = 0
    per_category: dict[str, list[bool]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source = expected_by_id[row["question_id"]]
        answer = str(source["answer"]).upper()
        is_correct = row["prediction"] == answer
        row["correct_answer"] = answer
        row["correct"] = is_correct
        correct += int(is_correct)
        category = str(source["question_category"])
        per_category[category].append(is_correct)
        confusion[answer][str(row["prediction"])] += 1

    scores = {
        "overall_accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "per_category": {
            category: {
                "accuracy": sum(values) / len(values),
                "correct": sum(values),
                "total": len(values),
            }
            for category, values in sorted(per_category.items())
        },
        "confusion": {
            answer: dict(sorted(predictions.items()))
            for answer, predictions in sorted(confusion.items())
        },
        "validation": problems,
    }
    submission = [
        {"question_id": row["question_id"], "prediction": row["prediction"]}
        for row in rows
    ]
    write_json(args.result_dir / "responses.json", rows)
    write_json(args.result_dir / "scores.json", scores)
    write_json(args.result_dir / "VideoQA.json", submission)
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(f"Validated submission: {args.result_dir / 'VideoQA.json'}")


if __name__ == "__main__":
    main()
