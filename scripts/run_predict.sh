#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="logs/eval/qwen3_full_sft"
mkdir -p "$log_dir"

llamafactory-cli train configs/qwen3_predict.yaml 2>&1 | tee "$log_dir/predict-$timestamp.log"
