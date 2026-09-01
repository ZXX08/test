#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-llamafactory-repro}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION="3.11.16"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment already exists: ${ENV_NAME}" >&2
  echo "Choose another name or remove the existing environment explicitly." >&2
  exit 2
fi

conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip setuptools wheel

conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 torchdata==0.11.0

conda run -n "${ENV_NAME}" python -m pip install \
  transformers==5.2.0 datasets==4.0.0 accelerate==1.11.0 \
  peft==0.18.1 trl==0.24.0 deepspeed==0.18.4 \
  av==16.0.0 decord==0.6.0 pyarrow==25.0.1 \
  numpy==2.4.6 pandas==2.3.3 pillow==11.3.0 \
  safetensors==0.8.0 sentencepiece==0.2.2 tiktoken==0.14.0 modelscope==1.39.1

conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_DIR}"

CUDA_HOME_PATH="$(conda run -n "${ENV_NAME}" python -c 'import site; print(site.getsitepackages()[0] + "/nvidia/cu13")' | tail -n 1)"
if [[ -x "${CUDA_HOME_PATH}/bin/nvcc" ]]; then
  conda env config vars set -n "${ENV_NAME}" CUDA_HOME="${CUDA_HOME_PATH}"
else
  echo "CUDA toolkit path was not found at ${CUDA_HOME_PATH}" >&2
  exit 3
fi

conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/verify_llamafactory_env.py"

echo "Environment created: ${ENV_NAME}"
echo "Activate with: conda activate ${ENV_NAME}"
