# Qwen2.5-0.5B 学习实验室（TRL + MLX）

> 🌐 [English Version → README.md](README.md)

一个完整可学习的工程，覆盖：
- 下载 `Qwen/Qwen2.5-0.5B`
- 理解数据格式与数据处理流水线
- 基于 Hugging Face TRL 做 LoRA 微调训练
- 基于 Apple MLX 框架做 LoRA 微调训练
- 两条路径的推理验证与对比

本工程以"学习为先"组织：每个脚本短小精悍，配置文件清晰可读，所有流程均有测试。

---

## 目录

1. [核心算法原理](#核心算法原理)
2. [项目结构](#项目结构)
3. [环境准备](#环境准备)
4. [运行测试](#运行测试)
5. [下载模型](#下载模型)
6. [数据准备](#数据准备)
7. [TRL LoRA 训练](#trl-lora-训练)
8. [MLX LoRA 训练](#mlx-lora-训练)
9. [两条学习路线](#两条学习路线)
10. [学习导向测试说明](#学习导向测试说明)
11. [常见问题排查](#常见问题排查)
12. [模型许可](#模型许可)

---

## 核心算法原理

### LoRA（低秩适配微调）

**背景问题**：全参数微调一个大模型需要海量显存（7B 模型 >100GB），个人机器无法承受。

**核心思想**：不直接修改原始权重矩阵 `W`（形状 `d×k`），而是在旁路插入两个小矩阵 `A`（`d×r`）和 `B`（`r×k`），秩 `r ≪ d`。

前向计算变为：

```
output = x @ (W + α/r · B @ A)
```

`W` 完全冻结，只训练 `A` 和 `B`。

**参数量对比（r=16 为例）**：

| 项目 | 参数量 |
|---|---|
| 原始权重 W（4096×4096） | 16M |
| LoRA 旁路 A+B | 约 131K |
| 节省比例 | **~99%** |

**关键超参数说明**：

| 参数 | 含义 | 推荐值 |
|---|---|---|
| `r` | 秩，控制 adapter 容量 | 4~64 |
| `alpha` | 缩放系数（实际缩放=alpha/r） | 通常设为 2r |
| `dropout` | 防过拟合 | 0.05 |
| `target_modules` | 插入 LoRA 的目标层 | `q_proj,v_proj` |

> 参考论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

### 指令微调格式（Instruction Tuning Format）

本项目使用 **Alpaca 三段式格式**，将原始数据映射为训练文本：

```
### Instruction:
<任务描述，告诉模型要做什么>

### Input:          ← 可选，有上下文时才写
<补充输入内容>

### Response:
<期望模型生成的标准答案>
```

**为什么格式一致性至关重要**：
- 训练和推理必须使用完全相同的格式，否则 adapter 泛化能力急剧下降
- 每种格式对应一种"提示词风格"，模型会记住该风格

---

### SFT 训练目标（监督微调）

SFT（Supervised Fine-Tuning）的损失函数是**下一词预测的交叉熵**：

```
Loss = -Σ log P(token_i | token_1...token_{i-1}, instruction)
```

关键训练技巧：

| 技术 | 原理 |
|---|---|
| 梯度累积 | 用多个小 batch 累加梯度，模拟大 batch，缓解显存不足 |
| 余弦学习率 | 训练末期平滑降低学习率，避免最后震荡 |
| Warmup | 训练初期从小学习率逐渐升到目标值，防止初期梯度爆炸 |

---

### MLX 与 Apple Silicon 统一内存

Apple MLX 框架的核心优势是 **统一内存（Unified Memory）**：

```
PyTorch/TRL：  CPU RAM ←→ (数据拷贝) ←→ GPU VRAM
MLX/Apple：    统一物理内存（CPU 和 GPU 直接共享，无拷贝开销）
```

**MLX vs TRL(MPS) 在 Apple Silicon 上的对比**：

| 对比维度 | MLX | TRL (MPS) |
|---|---|---|
| 内存利用率 | 高（统一内存） | 较低（需预留显存） |
| 小模型推理速度 | 快 | 中等 |
| 生态成熟度 | 成长中 | 成熟 |
| 调试便利性 | 较难 | 较方便 |

---

## 项目结构

```text
qwen25/
  configs/
    trl_sft.yaml              ← TRL 训练配置（最小数据集）
    trl_sft_downloaded.yaml   ← TRL 训练配置（下载数据集）
    mlx_lora.yaml             ← MLX LoRA 训练配置
  data/
    minimal/                  ← 内置 10 条样本，用于快速验证
      train.jsonl
      eval.jsonl
  scripts/
    demo_runner.py            ← 数据格式演示入口
  src/
    common/
      data.py                 ← 数据读取与转换（含 SFT 原理注释）
      download_dataset.py     ← 从 HuggingFace 下载训练集
      recommend_training.py   ← 按内存建议 batch/max-samples
      prompting.py            ← 指令模板（含格式原理注释）
    trl_path/
      download_model.py       ← 下载 Qwen2.5-0.5B
      train_sft.py            ← TRL + LoRA 训练主脚本（含原理注释）
      infer.py                ← TRL 路径推理
    mlx_path/
      download_model.py       ← 下载模型（MLX 路径）
      prepare_data.py         ← 转换为 MLX 格式
      train_lora.py           ← MLX LoRA 训练封装（含原理注释）
      infer.py                ← MLX 路径推理
  tests/                      ← 全部测试（含教学注释块）
  pyproject.toml
  requirements.txt
  requirements-trl.txt
  requirements-mlx.txt
  README.md                   ← 英文版文档
  README_zh.md                ← 中文版文档（本文件）
  README_beginner.md          ← 初学者路线（中文）
  README_advanced.md          ← 进阶者路线（中文）
```

---

## 环境准备

Python 3.10+ 推荐。

```bash
cd /Users/charles/Documents/AI/qwen25
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

安装 TRL 路径依赖：

```bash
python -m pip install -r requirements-trl.txt
```

安装 MLX 路径依赖（Apple Silicon）：

```bash
python -m pip install -r requirements-mlx.txt
```

---

## 运行测试

```bash
pytest -q
python scripts/demo_runner.py
```

---

## 下载模型

默认模型 ID 为 `Qwen/Qwen2.5-0.5B`。

```bash
# TRL 路径
python -m src.trl_path.download_model \
  --model-id Qwen/Qwen2.5-0.5B \
  --output-dir models/hf_qwen2_5_0_5b

# MLX 路径（可复用同一本地缓存）
python -m src.mlx_path.download_model \
  --model-id Qwen/Qwen2.5-0.5B \
  --output-dir models/hf_qwen2_5_0_5b
```

---

## 数据准备

训练数据格式为 JSONL，每行一条三元组样本：

```json
{"instruction": "翻译为中文", "input": "hello", "output": "你好"}
```

**先用内存建议工具估算合适的数据规模和 batch 大小**：

```bash
# 自动探测当前机器内存，给出 TRL 建议参数
python -m src.common.recommend_training --framework trl --profile balanced

# 手工指定 32GB 内存场景，输出 JSON 格式
python -m src.common.recommend_training --framework trl --memory-gb 32 --json

# 三种策略：safe（保守）/ balanced（平衡）/ aggressive（激进）
python -m src.common.recommend_training --framework trl --profile safe
```

**下载公开训练集**（以 Alpaca-cleaned 为例）：

```bash
python -m src.common.download_dataset \
  --dataset yahma/alpaca-cleaned \
  --split train \
  --max-samples 5000 \
  --eval-ratio 0.02 \
  --out-train data/downloaded/train.jsonl \
  --out-eval data/downloaded/eval.jsonl
```

如果数据集列名不同，用参数映射：

```bash
python -m src.common.download_dataset \
  --dataset some_org/some_dataset \
  --instruction-col prompt \
  --input-col context \
  --output-col response
```

---

## TRL LoRA 训练

**第一步：干跑验证配置（不加载模型）**

```bash
python -m src.trl_path.train_sft --config configs/trl_sft.yaml --dry-run
```

**第二步：最小数据集训练**

```bash
python -m src.trl_path.train_sft --config configs/trl_sft.yaml
```

**第三步：用下载数据集训练**

```bash
python -m src.trl_path.train_sft --config configs/trl_sft_downloaded.yaml
```

**断点续训（自动找最新 checkpoint）**：

```bash
python -m src.trl_path.train_sft \
  --config configs/trl_sft.yaml \
  --resume-from-checkpoint auto
```

**开启 W&B 实验追踪（可选）**：

```bash
python -m pip install wandb
python -m src.trl_path.train_sft \
  --config configs/trl_sft.yaml \
  --report-to wandb \
  --wandb-project qwen25-lab \
  --run-name qwen25-exp-01
```

**推理（叠加 LoRA adapter）**：

```bash
python -m src.trl_path.infer \
  --model-path Qwen/Qwen2.5-0.5B \
  --adapter-path outputs/trl_lora_adapter \
  --prompt "请用简短语言解释 LoRA 的作用。" \
  --max-new-tokens 120
```

---

## MLX LoRA 训练

`src/mlx_path/train_lora.py` 是 `mlx_lm.lora` 命令的配置化封装。

**干跑（只打印命令）**：

```bash
python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml --dry-run
```

封装脚本会自动将 `data/minimal/*.jsonl` 转换为 MLX 约定的 `train.jsonl` / `valid.jsonl`。  
如果数据已准备好，跳过转换：

```bash
python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml --skip-prepare-data
```

**实际训练**：

```bash
python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml
```

**推理（叠加 adapter）**：

```bash
python -m src.mlx_path.infer \
  --model-path Qwen/Qwen2.5-0.5B \
  --adapter-path outputs/mlx_adapters \
  --prompt "迁移学习的核心思想是什么？" \
  --max-tokens 120
```

---

## 两条学习路线

详细路线文档见：
- 初学者：[README_beginner.md](README_beginner.md)
- 进阶者：[README_advanced.md](README_advanced.md)

### 路线 A：初学者（最小可跑通闭环）

1. **理解数据格式**
   - 运行 `python scripts/demo_runner.py`
   - 阅读 `src/common/prompting.py` 和 `src/common/data.py` 顶部的原理注释
2. **安全干跑**
   - `python -m src.trl_path.train_sft --config configs/trl_sft.yaml --dry-run`
3. **最小 TRL LoRA 训练**
   - `python -m src.trl_path.train_sft --config configs/trl_sft.yaml`
   - 观察 `outputs/trl_lora_adapter` 下的产出物
4. **推理对比**
   - 分别用"有/无 adapter"执行推理，观察回答风格变化
5. **尝试 MLX 命令流**
   - `python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml --dry-run`

### 路线 B：进阶者（完整实验流程）

1. **用内存建议工具估算参数**
   - `python -m src.common.recommend_training --framework trl --profile balanced`
2. **下载更大的训练集**
   - `python -m src.common.download_dataset --dataset yahma/alpaca-cleaned ...`
3. **切换下载数据集配置训练**
   - `python -m src.trl_path.train_sft --config configs/trl_sft_downloaded.yaml`
4. **启用实验追踪与断点续训**
   - 使用 `--report-to wandb` + `--resume-from-checkpoint auto`
5. **TRL vs MLX 对比**
   - 两条路径均跑小规模训练，对比速度/内存/输出质量
6. **测试驱动迭代**
   - 改配置/模板前先扩充 `tests/`，保持 `pytest -q` 绿色

---

## 学习导向测试说明

每个测试文件前均有**中文教学注释块**，说明测试意图、关键断言与学习点。

| 测试文件 | 覆盖内容 |
|---|---|
| `test_data_pipeline.py` | JSONL 读取、空行跳过、缺字段校验 |
| `test_prompting.py` | 指令模板格式化与空白清理 |
| `test_dataset_download.py` | 列名映射、UTF-8 写出、参数校验 |
| `test_dataset_download_live.py` | 真实网络下载 E2E（环境变量开关） |
| `test_recommend_training.py` | 内存分档建议与配置片段生成 |
| `test_mlx_commands.py` | MLX 训练/推理命令构造 |
| `test_trl_train_utils.py` | 断点续训与日志后端解析 |
| `test_mlx_prepare.py` | MLX 数据格式转换 |

运行全部测试：

```bash
pytest -q
```

运行真实下载测试（需联网）：

```bash
RUN_REAL_DOWNLOAD_UT=1 pytest -q tests/test_dataset_download_live.py

# 控制下载样本量
RUN_REAL_DOWNLOAD_UT=1 RUN_REAL_DOWNLOAD_MAX_SAMPLES=500 pytest -q tests/test_dataset_download_live.py
```

---

## 常见问题排查

- **模型下载慢**：在 shell 里设置 HuggingFace 代理/镜像，或使用 `HF_ENDPOINT` 环境变量。
- **macOS 上训练慢**：没有 CUDA，TRL 走 CPU/MPS，batch 保持 1，`max_steps` 设小。
- **`mlx_lm` 命令找不到**：确认已安装 `requirements-mlx.txt` 且 Python 指向当前 venv。
- **OOM（内存不足）**：先运行 `recommend_training` 用 `safe` 策略，降低 `batch_size` 和 `max_seq_length`。

---

## 模型许可

使用前请查阅 HuggingFace 上 `Qwen/Qwen2.5-0.5B` 的模型卡片与许可协议，生产环境部署或再发布须遵守原始许可要求。

