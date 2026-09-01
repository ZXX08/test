import argparse
import os
import traceback
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer


try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None


DEFAULT_MODEL_PATH = ""
DEFAULT_DATA_DIR = ""
DEFAULT_OUTPUT_DIR = ""
OUTPUT_FILENAME = "Qwen3.5_sft_v2_output.csv"
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UrbanVideo-Bench local Qwen3.5 SFT inference")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Local HF model path")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="UrbanVideo-Bench dataset directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save inference CSV")
    parser.add_argument("--max_frames", type=int, default=32, help="Max uniformly sampled frames per video")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference: auto/cpu/cuda/cuda:0",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max generated tokens")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N pending samples for debugging")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def choose_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def get_pad_token_id(processor: AutoProcessor, tokenizer: AutoTokenizer) -> Optional[int]:
    processor_tokenizer = getattr(processor, "tokenizer", None)
    if processor_tokenizer is not None and processor_tokenizer.pad_token_id is not None:
        return processor_tokenizer.pad_token_id
    return tokenizer.pad_token_id


def print_model_metadata(
    config: AutoConfig,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
) -> None:
    print(f"config.model_type: {getattr(config, 'model_type', None)}")
    print(f"config.architectures: {getattr(config, 'architectures', None)}")
    print(f"processor class: {processor.__class__.__name__}")
    print(f"tokenizer class: {tokenizer.__class__.__name__}")
    print(f"model class: {model.__class__.__name__}")
    print(f"model has generate: {hasattr(model, 'generate')}")


def safe_load_model(
    model_path: str,
    device: str,
) -> Tuple[torch.nn.Module, AutoProcessor, AutoTokenizer, Optional[int]]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = choose_dtype(device)

    loaders = []
    if AutoModelForImageTextToText is not None:
        loaders.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
    if AutoModelForVision2Seq is not None:
        loaders.append(("AutoModelForVision2Seq", AutoModelForVision2Seq))
    loaders.append(("AutoModel", AutoModel))

    errors = []
    for loader_name, loader_cls in loaders:
        try:
            model = loader_cls.from_pretrained(
                model_path,
                dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if loader_name == "AutoModel" and not hasattr(model, "generate"):
                raise RuntimeError("loaded AutoModel does not provide generate()")
            model.to(device)
            model.eval()
            print_model_metadata(config=config, processor=processor, tokenizer=tokenizer, model=model)
            return model, processor, tokenizer, get_pad_token_id(processor, tokenizer)
        except Exception as e:
            errors.append(f"{loader_name}: {e}")

    raise RuntimeError(f"Failed to load model from {model_path}. Errors: {' | '.join(errors)}")


def build_output_path(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, OUTPUT_FILENAME)


def build_prompt(question_text: str) -> str:
    instruction = (
        "This video is provided as sampled frames from a first-person agent trajectory.\n"
        "Answer the multiple-choice question using the visual evidence.\n"
        "You must output exactly one option letter from A, B, C, D, E, F, G.\n"
        "Use this exact format:\n"
        "Option: A\n"
        "Do not include any other text."
    )
    return f"{instruction}\n\nQuestion:\n{question_text}"


def sample_video_frames(video_path: str, max_frames: int = 32) -> List[Image.Image]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file not found: {video_path}")
    if max_frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {max_frames}")

    vr = VideoReader(video_path, ctx=cpu(0))
    frame_count = len(vr)
    if frame_count <= 0:
        raise RuntimeError(f"invalid frame_count={frame_count} for video: {video_path}")

    sample_count = min(max_frames, frame_count)
    indices = np.linspace(0, frame_count - 1, num=sample_count, dtype=np.int64)
    indices = np.unique(indices).tolist()
    if len(indices) == 0:
        raise RuntimeError(f"failed to build frame indices for video: {video_path}")

    try:
        batch = vr.get_batch(indices).asnumpy()
    except Exception as e:
        raise RuntimeError(f"failed to decode sampled frames from video: {video_path}. {e}") from e

    if batch.size == 0:
        raise RuntimeError(f"no decodable sampled frames from video: {video_path}")

    frames = [Image.fromarray(frame).convert("RGB") for frame in batch]
    if len(frames) == 0:
        raise RuntimeError(f"no decodable sampled frames from video: {video_path}")
    return frames


def find_video_path(video_root: str, video_id: str) -> Optional[str]:
    # Some UrbanVideoBench archives unpack into videos/videos/.  Search both
    # layouts so callers do not need to rearrange a large video dataset.
    for root in (video_root, os.path.join(video_root, "videos")):
        candidate = os.path.join(root, str(video_id))
        if os.path.isfile(candidate):
            return candidate

        if not os.path.splitext(str(video_id))[1]:
            for ext in VIDEO_EXTENSIONS:
                path = candidate + ext
                if os.path.isfile(path):
                    return path

    return None


def apply_chat_template(processor: AutoProcessor, tokenizer: AutoTokenizer, messages: list) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise RuntimeError("Neither processor nor tokenizer provides apply_chat_template()")


def move_inputs_to_device(inputs, device: str):
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return inputs


def generate_one(
    model: torch.nn.Module,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    pad_token_id: Optional[int],
    question: str,
    frames: List[Image.Image],
    device: str,
    max_new_tokens: int,
) -> str:
    prompt = build_prompt(question)

    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = apply_chat_template(processor=processor, tokenizer=tokenizer, messages=messages)
    inputs = processor(
        text=[text],
        images=frames,
        return_tensors="pt",
        padding=True,
    )
    inputs = move_inputs_to_device(inputs, device)

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    if hasattr(inputs, "input_ids"):
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    else:
        generated_ids_trimmed = generated_ids

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def is_empty_output(value) -> bool:
    if pd.isna(value):
        return True
    output = str(value).strip()
    # Failed rows are retryable when resuming an existing output CSV.
    return output == "" or output.startswith("ERROR:")


def load_dataframe(data_parquet: str) -> pd.DataFrame:
    try:
        qa_df = pd.read_parquet(data_parquet)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet: {data_parquet}. "
            f"Please install parquet engine, e.g. `pip install pyarrow`. Original error: {e}"
        )

    required_columns = {"question", "video_id"}
    missing_columns = sorted(required_columns - set(qa_df.columns))
    if missing_columns:
        raise KeyError(f"MCQ.parquet must contain columns: {sorted(required_columns)}. Missing: {missing_columns}")
    return qa_df


def prepare_result_dataframe(qa_df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    res_df = qa_df.copy()
    if "Output" not in res_df.columns:
        res_df["Output"] = None
    if "Error" not in res_df.columns:
        res_df["Error"] = None

    if os.path.exists(output_csv):
        old_df = pd.read_csv(output_csv)
        if "Unnamed: 0" in old_df.columns:
            old_df = old_df.drop(columns=["Unnamed: 0"])
        if len(old_df) == len(res_df):
            if "Output" in old_df.columns:
                res_df["Output"] = old_df["Output"]
            if "Error" in old_df.columns:
                res_df["Error"] = old_df["Error"]
        else:
            print(f"Existing output length mismatch, ignoring resume file: {output_csv}")

    return res_df


def main() -> None:
    args = parse_args()

    data_parquet = os.path.join(args.data_dir, "MCQ.parquet")
    video_dir = os.path.join(args.data_dir, "videos")
    output_csv = build_output_path(args.output_dir)

    if not os.path.exists(data_parquet):
        raise FileNotFoundError(f"MCQ.parquet not found: {data_parquet}")
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"video directory not found: {video_dir}")

    qa_df = load_dataframe(data_parquet)
    res_df = prepare_result_dataframe(qa_df, output_csv)

    pending_idx = [i for i, value in enumerate(res_df["Output"].tolist()) if is_empty_output(value)]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError(f"limit must be >= 0, got {args.limit}")
        pending_idx = pending_idx[: args.limit]

    if len(pending_idx) == 0:
        print(f"All selected samples already have Output. Nothing to run. File: {output_csv}")
        return

    device = resolve_device(args.device)
    print(f"Loading model on device={device} from {args.model_path}")
    model, processor, tokenizer, pad_token_id = safe_load_model(args.model_path, device=device)

    print(f"Total samples: {len(res_df)}, selected pending: {len(pending_idx)}")
    for idx in tqdm(pending_idx, desc="UrbanVideo-Bench Qwen3.5 SFT inference"):
        question = str(res_df.at[idx, "question"])
        video_id = str(res_df.at[idx, "video_id"])

        try:
            video_path = find_video_path(video_dir, video_id)
            if video_path is None:
                error_message = f"ERROR: video not found for video_id={video_id} under {video_dir}"
                res_df.at[idx, "Output"] = error_message
                res_df.at[idx, "Error"] = error_message
            else:
                try:
                    frames = sample_video_frames(video_path, max_frames=args.max_frames)
                except Exception as e:
                    error_message = f"ERROR: failed to read video_id={video_id}. {e}"
                    res_df.at[idx, "Output"] = error_message
                    res_df.at[idx, "Error"] = error_message
                    res_df.to_csv(output_csv, index=False)
                    continue

                output_text = generate_one(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    pad_token_id=pad_token_id,
                    question=question,
                    frames=frames,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                res_df.at[idx, "Output"] = output_text
                res_df.at[idx, "Error"] = None

        except Exception as e:
            err_trace = traceback.format_exc(limit=1)
            error_message = f"ERROR: {e} | Trace: {err_trace.strip()}"
            res_df.at[idx, "Output"] = error_message
            res_df.at[idx, "Error"] = error_message

        res_df.to_csv(output_csv, index=False)

    print(f"Inference finished. Saved to: {output_csv}")


if __name__ == "__main__":
    main()
