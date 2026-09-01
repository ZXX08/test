# O3DVQA

Conda environment: `o3dvqa-test`, defined in
[`../envs/o3dvqa.yml`](../envs/o3dvqa.yml). Inference is local; the optional
evaluation step calls a configured OpenAI-compatible judge API.

Convert the grouped source JSON and validate every image path:

```bash
SOURCE_JSON=/path/to/O3DVQA_v2/test_qa.json \
IMAGE_ROOT=/path/to/O3DVQA_v2 \
OUTPUT_JSON=/path/to/Test_v2.json \
bash model_test/O3DVQA/run_convert.sh
```

Run a one-sample inference smoke test:

```bash
MODEL_PATH=/path/to/model \
DATA_JSON=/path/to/Test_v2.json \
DATA_ROOT=/path/to/O3DVQA_v2 \
RESULT_DIR=/path/to/output \
MODEL_NAME=qwen3vl_stage1_o3dvqa \
GPU=0 LIMIT=1 \
bash model_test/O3DVQA/run_inference.sh
```

Set `LIMIT=-1` for a complete inference. The response file is
`$RESULT_DIR/${MODEL_NAME}_responses.json`, and `--resume` skips completed IDs.


Never commit API keys or result files.


