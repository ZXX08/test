import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert grouped O3DVQA test QA data to inference format.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    image_root = Path(args.image_root).resolve()
    output = Path(args.output)
    grouped = json.loads(source.read_text(encoding="utf-8"))

    converted = []
    missing = []
    seen_ids = set()
    for dataset, scenes in grouped.items():
        for scene, samples in scenes.items():
            for sample in samples:
                sample_id = str(sample["id"])
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample id: {sample_id}")
                seen_ids.add(sample_id)

                image_info = sample.get("image_info") or {}
                relative_image = str(image_info.get("image_path") or sample.get("image_name", ""))
                relative_image = Path(relative_image.replace("\\", "/"))
                image_path = image_root / dataset / scene / relative_image
                if not image_path.is_file():
                    missing.append(str(image_path))

                conversations = sample.get("conversation") or sample.get("conversations") or []
                converted.append(
                    {
                        "id": sample_id,
                        "question_type": (sample.get("qa_info") or {}).get("type", ""),
                        "conversations": conversations,
                        "images": [str(image_path)],
                    }
                )

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"{len(missing)} image files are missing. First paths:\n{preview}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"Converted {len(converted)} samples to {output}")
    print(f"Validated {len(converted)} image paths under {image_root}")


if __name__ == "__main__":
    main()

