# VideoQA（UrbanVideoBench）8 卡测试复现

## 1. 目录与用途

本目录只负责 2026 ARTS 命题二的 VideoQA 赛道。完整流程为：环境与资源预检、
8 卡分片推理、结果合并与本地计分、提交格式校验、生成 `VideoQA.zip`。

主要文件：

- `run_8gpu.sh`：一键 8 卡入口。
- `infer_videoqa_rank.py`：每张 GPU 一个进程，均匀采样视频帧并推理。
- `preflight.py`：检查数据哈希、1071 个 ID、748 个视频、模型和 GPU。
- `merge_and_validate.py`：合并 8 个 rank，检查缺失、重复和非法答案。
- `package_submission.py`：生成根目录仅含 `VideoQA.json` 的 ZIP。
- `data/VideoQA_test.parquet`：官方测试标注的本地快照。

## 2. 环境配置

已验证环境：Ubuntu/Linux、Python 3.10、CUDA 12.6、8 × A100 40GB、
PyTorch 2.8.0+cu126、Transformers 5.2.0。

建议创建独立环境：

```bash
conda create -n arts-videoqa python=3.10 -y
conda activate arts-videoqa
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

验证关键包：

```bash
python -c "import torch, transformers, pyarrow, decord; print(torch.__version__, transformers.__version__)"
```

## 3. 测试数据路径

官方原始标注：

```text
/home/aiscuser/worspace-sj/-21-/VQA/test/UrbanVideoBench/VideoQA_test.parquet
```

本目录保存了一份完全相同的快照，默认运行使用：

```text
./data/VideoQA_test.parquet
```

两者 SHA256 均为：

```text
e28288b9c80824161fe2b2c846485a5901c4c4c463b5b9d3ffb6a0176f18f0c6
```

视频目录：

```text
/home/aiscuser/worspace-sj/-21-/VQA/test/UrbanVideoBench/videos
```

数据规模为 1071 道题、1071 个唯一 question ID、748 个唯一视频；选项支持 A–G。

## 4. 权重文件路径

默认使用当前表现最佳的匹配权重：

```text
/home/aiscuser/worspace-sj/model
```

该权重已测得 843/1071，准确率 78.7115%。Stage3 权重位于：

```text
/home/aiscuser/worspace-sj/stage3
```

其已测准确率为 722/1071，即 67.4136%。

模型目录至少应包含 `model.safetensors`、`config.json`、
`generation_config.json`、`processor_config.json`、`tokenizer.json` 和
`tokenizer_config.json`。

## 5. 测试代码运行流程与命令

先确认没有占卡程序或其他任务：

```bash
nvidia-smi
```

如果 `/blob/thinking.py` 正在占卡，先执行：

```bash
bash /home/aiscuser/workspace-ll/gpu_placeholder.sh stop
```

运行默认 Stage2 权重：

```bash
cd /home/aiscuser/worspace-sj/2026ARTS_official_eval/VideoQA
bash run_8gpu.sh
```

运行其他权重时必须设置新的 `RUN_NAME`：

```bash
MODEL_PATH=/path/to/new/model \
RUN_NAME=urbanvideo_new_model \
bash run_8gpu.sh
```

显式指定官方原始 parquet：

```bash
DATA_PARQUET=/home/aiscuser/worspace-sj/-21-/VQA/test/UrbanVideoBench/VideoQA_test.parquet \
MODEL_PATH=/home/aiscuser/worspace-sj/model \
RUN_NAME=urbanvideo_stage2_official_path \
bash run_8gpu.sh
```

可调参数及默认值：

```text
MAX_FRAMES=32
MAX_TOTAL_VIDEO_PIXELS=6291456
MIN_FREE_MIB=9000
PYTHON_BIN=python3
```

rank 结果支持断点续跑。同一 `RUN_NAME` 会复用已有 JSONL，因此更换权重、数据或
关键推理参数时必须更换 `RUN_NAME`。

## 6. 结果输出路径

```text
results/<RUN_NAME>/ranks/rank_00.jsonl ... rank_07.jsonl
results/<RUN_NAME>/responses.json
results/<RUN_NAME>/scores.json
results/<RUN_NAME>/VideoQA.json
results/<RUN_NAME>/VideoQA.zip
logs/<RUN_NAME>_preflight.log
logs/<RUN_NAME>_inference_8gpu.log
logs/<RUN_NAME>_merge.log
```

比赛提交文件是 `VideoQA.zip`。压缩包根目录必须只有 `VideoQA.json`，每条记录仅含：

```json
{
  "question_id": "2300",
  "prediction": "D"
}
```

目录中已归档两次参考运行：`results/urbanvideo_stage2_20260831/` 和
`results/urbanvideo_stage3_20260831/`。

