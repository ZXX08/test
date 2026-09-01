# UrbanVideoBench

Conda environment: `urbanvideo-test`, defined in
[`../envs/urbanvideobench.yml`](../envs/urbanvideobench.yml).

The data directory must contain:

```text
DATA_DIR/
├── MCQ.parquet
└── videos/
    └── *.mp4
```

The runner also supports archives that unpack as `videos/videos/*.mp4`.

Smoke test:

```bash
MODEL_PATH=/path/to/model \
DATA_DIR=/path/to/UrbanVideoBench \
OUTPUT_DIR=/path/to/output \
GPU=0 LIMIT=1 \
bash model_test/UrbanVideoBench/run_test.sh
```

Run all pending samples with `LIMIT=-1`. Existing blank or `ERROR:` outputs are
retried; successful rows are preserved. The output file is
`Qwen3.5_sft_v2_output.csv` and contains both `answer` (ground truth) and
`Output` (model prediction).


