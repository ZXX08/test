#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
log_dir="logs/training/qwen3_full_sft"
mkdir -p "$log_dir"

llamafactory-cli train configs/qwen3_full_sft.yaml 2>&1 | tee "$log_dir/run-$timestamp.log"
