from __future__ import annotations


# ============================================================
# 【原理说明】LoRA 微调 (Low-Rank Adaptation)
# ============================================================
#
# 背景问题：全参数微调一个 7B 模型需要 >100GB 显存，不适合个人机器。
#
# LoRA 核心思想：
#   不直接修改原始权重矩阵 W（形状 d×k），而是在旁路插入
#   两个小矩阵 A（d×r）和 B（r×k），其中秩 r << d。
#   前向计算变为：
#
#     output = x @ (W + α/r · B @ A)
#
#   W 冻结不变，只训练 A 和 B。
#
# 参数量对比（以 r=16 为例）：
#   - 原始权重：d × k（如 4096 × 4096 = 16M 参数）
#   - LoRA 旁路：d×r + r×k = 4096×16 + 16×4096 = 131K 参数
#   - 节省比例：约 99% 参数无需训练
#
# 关键超参数：
#   r       - 秩（rank）：控制 adapter 容量，通常 4~64，越大表达力越强
#   alpha   - 缩放系数：实际缩放比 = alpha/r，通常设为 2r
#   dropout - 防止 adapter 过拟合
#   target_modules - 插入 LoRA 的目标层，常见为 q_proj/v_proj
#
# 本脚本流程：
#   1. 加载 Qwen2.5-0.5B 基座模型（权重冻结）
#   2. 用 PEFT 库在注意力层插入 LoRA adapter
#   3. 用 TRL 的 SFTTrainer 在指令数据上训练 adapter
#   4. 保存 adapter 权重（几十 MB），推理时叠加到基座上使用
#
# 参考：LoRA 原论文 https://arxiv.org/abs/2106.09685
# ============================================================
#
# ============================================================
# 【原理说明】SFTTrainer 与 Causal LM 训练目标
# ============================================================
#
# SFTTrainer（Supervised Fine-Tuning Trainer）是 TRL 库对
# HuggingFace Trainer 的封装，专为指令微调场景优化：
#
#   训练目标：Next Token Prediction（下一词预测）
#   损失函数：交叉熵 Loss = -Σ log P(y_t | x, y_1...y_{t-1})
#
# 梯度累积（gradient_accumulation_steps）：
#   在显存受限时，用多个小 batch 的梯度累加来模拟大 batch 训练，
#   等效全局 batch = per_device_batch × accumulation_steps
#
# 余弦学习率调度（cosine schedule）：
#   学习率从初始值按余弦曲线下降到接近 0，在训练末期平滑收尾，
#   避免最后几步的震荡影响模型最终性能。
# ============================================================

import argparse
import os
from pathlib import Path

import yaml

from src.common.data import build_text_records, read_jsonl


def parse_args() -> argparse.Namespace:
    """解析训练入口参数，支持实验覆盖配置项。"""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with TRL SFT + LoRA.")
    parser.add_argument("--config", default="configs/trl_sft.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and data only.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path or `auto`. Overrides config when provided.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run name override.")
    parser.add_argument(
        "--report-to",
        default=None,
        help="Logging backend: none | wandb | tensorboard | comma-separated list.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置文件。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(file_path: str | Path) -> list[dict]:
    """加载并转换为 TRL SFTTrainer 需要的 text 样本列表。"""
    rows = read_jsonl(file_path)
    return build_text_records(rows)


def resolve_report_to(cli_value: str | None, cfg: dict) -> str | list[str]:
    """根据 CLI 与配置确定日志上报后端。"""
    if cli_value is not None:
        raw = cli_value
    else:
        raw = str(cfg.get("experiment", {}).get("report_to", "none"))

    if raw == "none":
        return "none"

    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts or "none"

    return raw


def _checkpoint_step(path: Path) -> int:
    """从 checkpoint-xxx 目录名中提取步数。"""
    try:
        return int(path.name.rsplit("-", maxsplit=1)[-1])
    except (IndexError, ValueError):
        return -1


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    """扫描输出目录，返回最新 checkpoint 路径。"""
    out = Path(output_dir)
    if not out.exists():
        return None
    candidates = [p for p in out.glob("checkpoint-*") if p.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=_checkpoint_step)
    return str(latest)


def resolve_resume_checkpoint(cli_value: str | None, cfg: dict) -> str | None:
    """解析断点续训来源：none / auto / 指定路径。"""
    training_cfg = cfg.get("training", {})
    raw = cli_value if cli_value is not None else training_cfg.get("resume_from_checkpoint", "none")

    if raw in (None, "", "none"):
        return None
    if raw == "auto":
        return find_latest_checkpoint(training_cfg["output_dir"])
    return str(raw)


def configure_wandb_env(args: argparse.Namespace, cfg: dict) -> None:
    """在启用 wandb 时注入必要环境变量。"""
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    elif cfg.get("experiment", {}).get("wandb_project"):
        os.environ["WANDB_PROJECT"] = str(cfg["experiment"]["wandb_project"])

    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    elif cfg.get("experiment", {}).get("wandb_entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["experiment"]["wandb_entity"])

    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name


def main() -> None:
    """执行 TRL + LoRA 监督微调流程。"""
    args = parse_args()
    cfg = load_yaml(args.config)

    train_ds = build_dataset(cfg["data"]["train_file"])
    eval_ds = build_dataset(cfg["data"]["eval_file"])

    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples:  {len(eval_ds)}")

    resolved_report_to = resolve_report_to(args.report_to, cfg)
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, cfg)
    run_name = args.run_name or str(cfg.get("experiment", {}).get("run_name", "qwen25-trl-sft"))

    print(f"Report to:     {resolved_report_to}")
    print(f"Run name:      {run_name}")
    print(f"Resume from:   {resume_checkpoint or 'none'}")

    if args.dry_run:
        # dry-run 仅校验配置与数据，不触发模型下载和训练。
        print("Dry run complete. Config and dataset format are valid.")
        return

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    train_ds = Dataset.from_list(train_ds)
    eval_ds = Dataset.from_list(eval_ds)

    model_name = cfg["model"]["model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        # Qwen 某些 tokenizer 无 pad_token，统一回退到 eos_token。
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        torch_dtype="auto",
    )

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )

    training_cfg = cfg["training"]
    logging_dir = training_cfg.get("logging_dir", str(Path(training_cfg["output_dir"]) / "logs"))

    if resolved_report_to == "wandb" or (
        isinstance(resolved_report_to, list) and "wandb" in resolved_report_to
    ):
        configure_wandb_env(args, cfg)

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        run_name=run_name,
        logging_dir=logging_dir,
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        logging_steps=int(training_cfg["logging_steps"]),
        eval_steps=int(training_cfg["eval_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        weight_decay=float(training_cfg["weight_decay"]),
        lr_scheduler_type=str(training_cfg["lr_scheduler_type"]),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(training_cfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(training_cfg.get("greater_is_better", False)),
        seed=int(training_cfg.get("seed", 42)),
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=resolved_report_to,
        fp16=bool(cfg["model"].get("use_fp16", False)),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=int(cfg["data"]["max_seq_length"]),
        args=training_args,
        peft_config=peft_config,
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    # 保存 adapter 与 tokenizer，方便后续推理或继续训练。
    trainer.save_model(training_cfg["output_dir"])
    tokenizer.save_pretrained(training_cfg["output_dir"])
    print(f"Saved adapter artifacts to: {training_cfg['output_dir']}")


if __name__ == "__main__":
    main()


