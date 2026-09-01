import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig


DEFAULT_DATA_JSON = "/root/workspace/O3DVQA/Test.json"
DEFAULT_DATA_ROOT = "/root/workspace/O3DVQA/O3DVQA"
DEFAULT_MODEL_NAME = "stage2"
DEFAULT_MODEL_PATH = "/root/workspace/models/stage2"
SYSTEM_PROMPT = (
    "You are an assistant who perfectly answer question in urban environment. "
    "Only based on the image, you should directly answer the height, width, volume and distance question "
    "with exact number. Answer the distance without output intermediate process. You should answer the "
    "direction question in the direction of the clock with taking your front as 12 o'clock, your left as "
    "9 o'clock, and your right as 3 o'clock."
)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def get_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_model(model_path: str):
    print(f"Loading tested model from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"config.model_type: {getattr(config, 'model_type', None)}")
    print(f"config.architectures: {getattr(config, 'architectures', None)}")
    print(f"processor class: {processor.__class__.__name__}")
    print(f"tokenizer class: {tokenizer.__class__.__name__}")

    model_classes = []
    for cls_name in ["AutoModelForImageTextToText", "AutoModelForVision2Seq"]:
        try:
            module = __import__("transformers", fromlist=[cls_name])
            model_classes.append(getattr(module, cls_name))
        except Exception:
            pass
    model_classes.append(AutoModel)

    last_error = None
    for model_cls in model_classes:
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
            if not hasattr(model, "generate"):
                raise RuntimeError(f"{model_cls.__name__} loaded a model without generate().")
            print(f"model class: {model.__class__.__name__}")
            return model, processor, tokenizer
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load model from {model_path}: {last_error}")


def normalize_windows_path(path: str) -> str:
    return path.replace("\\", "/")


def resolve_image_path(item: dict[str, Any], data_root: str, dataset: str | None = None, scene: str | None = None) -> str:
    images = item.get("images")
    if isinstance(images, list) and images:
        raw_path = images[0]
    else:
        image_info = item.get("image_info") or {}
        raw_path = image_info.get("image_path") or item.get("image_name")

    if not raw_path:
        raise ValueError(f"Sample has no image path: {item.get('id')}")

    normalized = normalize_windows_path(str(raw_path))
    if os.path.exists(normalized):
        return normalized

    marker = "/O3DVQA/"
    if marker in normalized:
        rel_path = normalized.split(marker, 1)[1]
        candidate = os.path.join(data_root, rel_path)
        if os.path.exists(candidate):
            return candidate

    if dataset and scene:
        candidate = os.path.join(data_root, dataset, scene, normalized)
        if os.path.exists(candidate):
            return candidate

    candidate = os.path.join(data_root, normalized)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f"Image file not found: {raw_path}. Tried under data root: {data_root}")


def iter_samples(data: Any):
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield None, None, item
        return

    if isinstance(data, dict):
        for dataset, scenes in data.items():
            if not isinstance(scenes, dict):
                continue
            for scene, entries in scenes.items():
                if not isinstance(entries, list):
                    continue
                for item in entries:
                    if isinstance(item, dict):
                        yield dataset, scene, item
        return

    raise TypeError(f"Unsupported JSON root type: {type(data)}")


def extract_question(item: dict[str, Any]) -> str:
    if item.get("query_question"):
        return str(item["query_question"])

    conversations = item.get("conversations") or item.get("conversation") or []
    if conversations and isinstance(conversations[0], dict):
        question = str(conversations[0].get("value", ""))
        return question.replace("<image>", "").strip()
    return ""


def extract_answer(item: dict[str, Any]) -> str:
    conversations = item.get("conversations") or item.get("conversation") or []
    if len(conversations) > 1 and isinstance(conversations[1], dict):
        return str(conversations[1].get("value", ""))
    return ""


def infer_one(model, processor, tokenizer, image_path: str, question: str, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = get_device(model)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    decoder = processor if hasattr(processor, "batch_decode") else tokenizer
    return decoder.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def result_bucket(item: dict[str, Any], dataset: str | None, scene: str | None, image_path: str) -> tuple[str, str]:
    if dataset and scene:
        return dataset, scene

    raw_path = normalize_windows_path(image_path)
    parts = raw_path.split("/O3DVQA/", 1)[-1].split("/")
    if parts and parts[0] == "O3DVQA":
        parts = parts[1:]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "O3DVQA", "test"


def load_existing(path: str | Path) -> tuple[defaultdict, set[str]]:
    responses = defaultdict(lambda: defaultdict(list))
    done_ids = set()
    if not Path(path).exists():
        return responses, done_ids

    data = read_json(path)
    for dataset, scenes in data.items():
        for scene, items in scenes.items():
            for item in items:
                responses[dataset][scene].append(item)
                if item.get("id") is not None:
                    done_ids.add(str(item["id"]))
    return responses, done_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Open3D-VQA inference.")
    parser.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--result-dir", default="only_o3dvqa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = read_json(args.data_json)
    result_path = Path(args.result_dir) / f"{args.model_name}_responses.json"
    responses, done_ids = load_existing(result_path) if args.resume else (defaultdict(lambda: defaultdict(list)), set())
    model, processor, tokenizer = load_model(args.model_path)

    processed = 0
    samples = list(iter_samples(data))
    for dataset, scene, item in tqdm(samples, desc="Processing"):
        item_id = str(item.get("id"))
        if args.resume and item_id in done_ids:
            continue
        if args.limit > 0 and processed >= args.limit:
            break

        image_path = resolve_image_path(item, args.data_root, dataset, scene)
        question = extract_question(item)
        answer = extract_answer(item)
        bucket_dataset, bucket_scene = result_bucket(item, dataset, scene, image_path)

        response_text = infer_one(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            image_path=image_path,
            question=question,
            max_new_tokens=args.max_new_tokens,
        )

        response = {
            "id": item.get("id"),
            "question_type": item.get("question_type", ""),
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "qa_info": item.get("qa_info", {}),
            "question": question,
            "answer": answer,
            "response": response_text,
            "model_path": args.model_path,
        }
        responses[bucket_dataset][bucket_scene].append(response)
        processed += 1

        if processed % args.save_every == 0:
            save_json(result_path, {k: dict(v) for k, v in responses.items()})

    save_json(result_path, {k: dict(v) for k, v in responses.items()})
    print(f"Saved {processed} new responses to {result_path}")


if __name__ == "__main__":
    main()

