# Qwen2.5 学习路线（进阶者）

本路线目标：构建可复现实验流程，覆盖数据扩展、日志追踪、断点续训与框架对比。

## 1) 安装完整依赖

```bash
cd /Users/charles/Documents/AI/qwen25
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements-trl.txt
python -m pip install -r requirements-mlx.txt
```

## 2) 下载并切分训练集

```bash
python -m src.common.download_dataset \
  --dataset yahma/alpaca-cleaned \
  --split train \
  --max-samples 5000 \
  --eval-ratio 0.02 \
  --out-train data/downloaded/train.jsonl \
  --out-eval data/downloaded/eval.jsonl
```

先用内存建议工具给出可行的规模与 batch：

```bash
python -m src.common.recommend_training --framework trl --profile balanced --json
python -m src.common.recommend_training --framework mlx --profile aggressive --json
```

## 3) TRL 实验训练（可续训）

```bash
python -m src.trl_path.train_sft --config configs/trl_sft_downloaded.yaml
python -m src.trl_path.train_sft --config configs/trl_sft_downloaded.yaml --resume-from-checkpoint auto
```

可选 W&B 追踪：

```bash
python -m src.trl_path.train_sft \
  --config configs/trl_sft_downloaded.yaml \
  --report-to wandb \
  --wandb-project qwen25-lab \
  --run-name exp-downloaded-01
```

## 4) MLX 路线对比

```bash
python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml --dry-run
python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml
```

## 5) 验证与回归

```bash
pytest -q
```

建议重点看：
- `tests/test_trl_train_utils.py`
- `tests/test_mlx_commands.py`
- `tests/test_dataset_download.py`

## 6) 输出物检查清单

- `outputs/trl_lora_adapter_downloaded` 或 `outputs/trl_lora_adapter` 存在
- `data/downloaded/train.jsonl` 与 `data/downloaded/eval.jsonl` 样本数合理
- 能完成至少一次 checkpoint 恢复训练
- 能给出 TRL 与 MLX 的速度/资源对比结论

