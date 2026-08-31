$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = "logs/training/qwen3_full_sft"
New-Item -ItemType Directory -Force $logDir | Out-Null

llamafactory-cli train configs/qwen3_full_sft.yaml 2>&1 |
  Tee-Object -FilePath "$logDir/run-$timestamp.log"
