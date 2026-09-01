#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV:-o3dvqa-test}"
SOURCE_JSON="${SOURCE_JSON:?Set SOURCE_JSON to the grouped test_qa.json file}"
IMAGE_ROOT="${IMAGE_ROOT:?Set IMAGE_ROOT to the O3DVQA_v2 image root}"
OUTPUT_JSON="${OUTPUT_JSON:?Set OUTPUT_JSON to the converted JSON path}"

exec conda run --no-capture-output -n "${ENV_NAME}" python "${SCRIPT_DIR}/convert_test_qa.py" \
  --source "${SOURCE_JSON}" \
  --image-root "${IMAGE_ROOT}" \
  --output "${OUTPUT_JSON}" \
  "$@"

