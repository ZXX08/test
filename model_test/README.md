# Model test toolkit

Repository: https://github.com/ZXX08/test

This directory contains the model-evaluation code collected from the local
`VQA/test` workspace. Large datasets, videos, model checkpoints, caches, and
generated results are intentionally excluded from Git.

## Test suites

| Suite | Input | Entry point | Conda environment |
| --- | --- | --- | --- |
| [AirCopBench](AirCopBench/README.md) | Four annotation JSON files and multi-UAV images | `AirCopBench/run_test.sh` | `aircopbench-test` |
| [O3DVQA](O3DVQA/README.md) | O3DVQA JSON and RGB images | `O3DVQA/run_inference.sh` | `o3dvqa-test` |
| [UrbanVideoBench](UrbanVideoBench/README.md) | `MCQ.parquet` and videos | `UrbanVideoBench/run_test.sh` | `urbanvideo-test` |

## Create environments

Run from the repository root:

```bash
conda env create -f model_test/envs/aircopbench.yml
conda env create -f model_test/envs/o3dvqa.yml
conda env create -f model_test/envs/urbanvideobench.yml
```

Each runner activates its own Conda environment and accepts additional Python
arguments after the environment-variable configuration documented in the suite
README. Use `LIMIT=1` for a smoke test before a complete run.


