# Qwen3 Full SFT

本仓库用于 Qwen3 全参数 SFT 训练、测试和结果复现。

仓库地址：https://github.com/ZXX08/test

完整 LLaMA-Factory 训练代码快照位于 [`trian/`](trian/README_zh.md)。

## 模型测试工具

AirCopBench、O3DVQA 和 UrbanVideoBench 的测试代码已统一整理到
[`model_test/`](model_test/README.md)。该目录为每个测试任务提供：

- 独立的 Conda 环境配置；
- 数据和模型路径说明；
- 单样本 smoke test 与完整推理脚本；
- 断点续跑和输出格式说明。

视频、图片、模型权重、缓存和推理结果不提交到 Git，请按照各测试目录的
README 配置本地路径。

## 环境配置

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install tensorboard
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如使用 CUDA，请先安装与显卡驱动匹配的 PyTorch。

## 完整训练

训练配置：

```text
configs/qwen3_full_sft.yaml
```

训练命令：

```bash
llamafactory-cli train configs/qwen3_full_sft.yaml
```

Windows 可直接运行：

```powershell
.\scripts\run_full_train.ps1
```

关键超参数：

```text
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
dataset: identity,alpaca_en_demo
finetuning_type: full
cutoff_len: 2048
val_size: 0.05
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
eval_strategy: epoch
save_strategy: epoch
```

## 日志和权重

```text
训练日志: logs/training/qwen3_full_sft/
验证记录: models/qwen3-4b/full/sft/trainer_state.json
权重路径: models/qwen3-4b/full/sft/
TensorBoard: tensorboard --logdir logs/training/qwen3_full_sft
```

训练完成后重点保留：

```text
models/qwen3-4b/full/sft/config.json
models/qwen3-4b/full/sft/tokenizer.json
models/qwen3-4b/full/sft/model*.safetensors
models/qwen3-4b/full/sft/trainer_state.json
models/qwen3-4b/full/sft/training_loss.png
```

## 测试

测试配置：

```text
configs/qwen3_predict.yaml
```

测试命令：

```bash
llamafactory-cli train configs/qwen3_predict.yaml
```

Windows 可直接运行：

```powershell
.\scripts\run_predict.ps1
```

测试相关路径：

```text
测试数据: data/alpaca_en_demo.json
权重路径: models/qwen3-4b/full/sft/
测试日志: logs/eval/qwen3_full_sft/
结果输出: outputs/predictions/qwen3_full_sft/
```

## 权重下载和使用

仓库不提交大体积权重。将训练好的权重下载或复制到：

```text
models/qwen3-4b/full/sft/
```

然后运行测试命令即可加载该权重。
