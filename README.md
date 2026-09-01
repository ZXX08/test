# Qwen3.5 训练与测试

## 一、训练

训练代码位于 `trian/`，需要在该目录下安装和启动。

### 1. 配置训练环境

```bash
cd trian
conda env create -f environment/llamafactory-core.yml

conda run -n llamafactory-repro \
    python scripts/verify_llamafactory_env.py
```
或者使用
```bash
cd train
 bash scripts/create_llamafactory_env.sh llamafactory-repro
```

如使用 GPU，请先安装与 CUDA/驱动匹配的 PyTorch。

### 2. 第一阶段训练

第一阶段配置文件：

```text
trian/examples/train_full/qwen3.5_stage1.yaml
```

启动命令：

```bash
cd trian
llamafactory-cli train examples/train_full/qwen3.5_stage1.yaml
```

第一阶段输出权重：

```text
/root/workspace-sj/saves/qwen3vl_stage1_aircopbench_o3dvqa
```

### 3. 第二阶段训练

第一阶段训练完成后，再启动第二阶段训练。

第二阶段配置文件：

```text
trian/examples/train_full/qwen3.5_stage2.yaml
```

启动命令：

```bash
cd trian
llamafactory-cli train examples/train_full/qwen3.5_stage2.yaml
```

第二阶段默认读取第一阶段权重：

```text
/root/workspace-sj/saves/qwen3vl_stage1_aircopbench_o3dvqa
```

第二阶段输出权重：

```text
/root/workspace-sj/saves/qwen3vl_stage2_urbanvideobench_trian
```

## 二、测试

测试代码位于 `model_test/`。三个测试集分别配置环境并运行。

### 1. O3DQA 测试

O3DQA 使用第一阶段训练结果：

```text
MODEL_PATH=/root/workspace-sj/saves/qwen3vl_stage1_aircopbench_o3dvqa
```

配置环境：

```bash
conda env create -f model_test/envs/o3dvqa.yml
```

启动测试：

```bash
MODEL_PATH=/root/workspace-sj/saves/qwen3vl_stage1_aircopbench_o3dvqa \
DATA_JSON=/path/to/Test_v2.json \
DATA_ROOT=/path/to/O3DVQA_v2 \
RESULT_DIR=/path/to/output/o3dvqa \
MODEL_NAME=qwen35_stage1 \
GPU=0 LIMIT=-1 \
bash model_test/O3DVQA/run_inference.sh
```

可选评测：

```bash
RESULT_PATH=/path/to/output/o3dvqa/qwen35_stage1_responses.json \
DATA_JSON=/path/to/Test_v2.json \
OPENAI_API_KEY=your-key \
bash model_test/O3DVQA/run_evaluation.sh
```

### 2. AirCopBench 测试

AirCopBench 使用第一阶段训练结果：

```text
MODEL_PATH=/root/workspace-sj/saves/qwen3vl_stage1_aircopbench_o3dvqa
```

配置环境：

```bash
conda env create -f model_test/envs/aircopbench.yml
```

启动测试：

```bash
MODEL_PATH=/root/workspace-sj/saves/qwen3vl_stage1_aircopbench_o3dvqa \
ANNOTATIONS_DIR=model_test/AircopBench \
IMAGES_ROOT=/path/to/AirCopBench \
OUTPUT_DIR=/path/to/output/aircopbench \
GPU=0 LIMIT=-1 \
bash model_test/AircopBench/run_test.sh
```

### 3. UrbanVideoBench 测试

UrbanVideoBench 使用第二阶段训练结果：

```text
MODEL_PATH=/root/workspace-sj/saves/qwen3vl_stage2_urbanvideobench_trian
```

配置环境：

```bash
conda env create -f model_test/envs/urbanvideobench.yml
```

启动测试：

```bash
MODEL_PATH=/root/workspace-sj/saves/qwen3vl_stage2_urbanvideobench_trian \
DATA_DIR=/path/to/UrbanVideoBench \
OUTPUT_DIR=/path/to/output/urbanvideobench \
GPU=0 LIMIT=-1 \
bash model_test/UrbanVideoBench/run_test.sh
```

测试前可先将 `LIMIT=-1` 改为 `LIMIT=1` 做单样本检查。
