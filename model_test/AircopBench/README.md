# AirCopBench

Conda environment: `aircopbench-test`, defined in
[`../envs/aircopbench.yml`](../envs/aircopbench.yml).

Required data:

- `Real2_VQA_test.json`, `Sim3_VQA_test.json`, `Sim5_VQA_test.json`, and `Sim6_VQA_test.json`
- Extracted `AirCopBench/Real_2_UAVs`, `Sim_3_UAVs`, `Sim_5_UAVs`, and `Sim_6_UAVs` image trees
- A local Hugging Face-compatible Qwen vision-language checkpoint

Smoke test:

```bash
MODEL_PATH=/path/to/model \
ANNOTATIONS_DIR=/path/to/annotations \
IMAGES_ROOT=/path/to/AircopBench \
OUTPUT_DIR=/path/to/output \
GPU=0 LIMIT=1 \
bash model_test/AircopBench/run_test.sh
```

Remove `LIMIT=1` or set `LIMIT=-1` for the complete evaluation. The runner
always enables `--resume`; completed sample IDs in `results.jsonl` are skipped.
Outputs include row-level `results.json`/`results.jsonl` and aggregate
`scores.json`.


