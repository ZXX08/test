#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-o3dvqa-test}"
RESULT_PATH="${RESULT_PATH:?Set RESULT_PATH to the inference responses JSON}"
DATA_JSON="${DATA_JSON:?Set DATA_JSON to the converted O3DVQA test JSON}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/evaluation_output}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"
BASE_URL="${BASE_URL:-https://www.autodl.art/api/v1}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"

if [[ -z "${API_KEY}" ]]; then
  echo "Set API_KEY or OPENAI_API_KEY for judge-based evaluation." >&2
  exit 1
fi

exec conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT_DIR}/evaluation.py" \
  --result-path "${RESULT_PATH}" \
  --data-json "${DATA_JSON}" \
  --judge-model-name "${JUDGE_MODEL}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --log-dir "${LOG_DIR}" \
  --resume \
  "$@"

