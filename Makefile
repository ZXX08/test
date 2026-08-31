.PHONY: build train predict

RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")
BUILD := $(shell command -v uv >/dev/null 2>&1 && echo "uv build" || echo "python -m build")

build:
	$(BUILD)

train:
	$(RUN) llamafactory-cli train configs/qwen3_full_sft.yaml

predict:
	$(RUN) llamafactory-cli train configs/qwen3_predict.yaml
