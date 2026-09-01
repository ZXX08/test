#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf-cache}"
export TOKENIZERS_PARALLELISM=false

DATA_PARQUET="${DATA_PARQUET:-${ROOT_DIR}/data/VideoQA_test.parquet}"
VIDEO_ROOT="${VIDEO_ROOT:-/home/aiscuser/worspace-sj/-21-/VQA/test/UrbanVideoBench/videos}"
MODEL_PATH="${MODEL_PATH:-/home/aiscuser/worspace-sj/model}"
RUN_NAME="${RUN_NAME:-urbanvideo_stage2_reproduce}"
RESULT_DIR="${ROOT_DIR}/results/${RUN_NAME}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_TOTAL_VIDEO_PIXELS="${MAX_TOTAL_VIDEO_PIXELS:-6291456}"

mkdir -p "${ROOT_DIR}/logs" "${RESULT_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/preflight.py" \
  --data-parquet "${DATA_PARQUET}" \
  --video-root "${VIDEO_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --min-free-mib "${MIN_FREE_MIB}" \
  2>&1 | tee "${ROOT_DIR}/logs/${RUN_NAME}_preflight.log"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${ROOT_DIR}/infer_videoqa_rank.py" \
  --data-parquet "${DATA_PARQUET}" \
  --video-root "${VIDEO_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --result-dir "${RESULT_DIR}" \
  --max-frames "${MAX_FRAMES}" \
  --max-total-video-pixels "${MAX_TOTAL_VIDEO_PIXELS}" \
  --max-new-tokens 16 \
  2>&1 | tee "${ROOT_DIR}/logs/${RUN_NAME}_inference_8gpu.log"

"${PYTHON_BIN}" "${ROOT_DIR}/merge_and_validate.py" \
  --data-parquet "${DATA_PARQUET}" \
  --result-dir "${RESULT_DIR}" \
  2>&1 | tee "${ROOT_DIR}/logs/${RUN_NAME}_merge.log"

"${PYTHON_BIN}" "${ROOT_DIR}/package_submission.py" \
  --input "${RESULT_DIR}/VideoQA.json" \
  --output "${RESULT_DIR}/VideoQA.zip"

echo "VideoQA completed: ${RESULT_DIR}/VideoQA.zip"

