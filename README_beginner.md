# Qwen2.5 学习路线（初学者）

本路线目标：用最小成本跑通一次“数据 -> 训练 -> 推理”的完整闭环。

## 1) 环境准备

```bash
cd /Users/charles/Documents/AI/qwen25
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements-trl.txt
```

## 2) 先看数据和模板

```bash
python scripts/demo_runner.py
pytest -q tests/test_data_pipeline.py tests/test_prompting.py
```

## 3) TRL 干跑（不训练）

```bash
python -m src.trl_path.train_sft --config configs/trl_sft.yaml --dry-run
```

可先用内存建议工具估算参数：

```bash
python -m src.common.recommend_training --framework trl --profile safe
```

## 4) 进行一次最小训练

```bash
python -m src.trl_path.train_sft --config configs/trl_sft.yaml
```

## 5) 推理验证

```bash
python -m src.trl_path.infer \
  --model-path Qwen/Qwen2.5-0.5B \
  --adapter-path outputs/trl_lora_adapter \
  --prompt "请用简短语言解释 LoRA 的作用。" \
  --max-new-tokens 120
```

## 6) 完成标准

- 能解释 `instruction/input/output` 如何转成训练 `text`
- 能独立跑通 `--dry-run` 与一次小训练
- 能使用 adapter 做推理并看到结果变化

