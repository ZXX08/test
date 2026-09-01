$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = "logs/eval/qwen3_full_sft"
New-Item -ItemType Directory -Force $logDir | Out-Null

llamafactory-cli train configs/qwen3_predict.yaml 2>&1 |
  Tee-Object -FilePath "$logDir/predict-$timestamp.log"
