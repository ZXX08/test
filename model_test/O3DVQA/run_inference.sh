#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-o3dvqa-test}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Hugging Face model directory}"
DATA_JSON="${DATA_JSON:?Set DATA_JSON to converted O3DVQA test JSON}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the O3DVQA_v2 image root}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/output}"
MODEL_NAME="${MODEL_NAME:-model}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:--1}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"

exec conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT_DIR}/inference.py" \
  --data-json "${DATA_JSON}" \
  --data-root "${DATA_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --result-dir "${RESULT_DIR}" \
  --limit "${LIMIT}" \
  --resume \
  "$@"


