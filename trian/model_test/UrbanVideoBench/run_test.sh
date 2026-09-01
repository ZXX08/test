#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-urbanvideo-test}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Hugging Face model directory}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the directory containing MCQ.parquet and videos/}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:--1}"
MAX_FRAMES="${MAX_FRAMES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"

ARGS=(
  --model_path "${MODEL_PATH}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --device cuda:0
  --max_frames "${MAX_FRAMES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)
if [[ "${LIMIT}" != "-1" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

exec conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT_DIR}/run_qwenv2.py" "${ARGS[@]}" "$@"

