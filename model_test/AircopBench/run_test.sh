#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-aircopbench-test}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Hugging Face model directory}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:?Set ANNOTATIONS_DIR to the directory containing the four *_VQA_test.json files}"
IMAGES_ROOT="${IMAGES_ROOT:?Set IMAGES_ROOT to the extracted AirCopBench directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:--1}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"

exec conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT_DIR}/run_aircopbench.py" \
  --annotations-dir "${ANNOTATIONS_DIR}" \
  --images-root "${IMAGES_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --resume \
  "$@"


