from __future__ import annotations


# ============================================================
# 【原理说明】DPO 直接偏好优化 (Direct Preference Optimization)
# ============================================================
#
# RLHF 的传统路径（PPO）需要同时维护四个模型：
#   策略模型、参考模型、奖励模型、价值模型，训练极其复杂。
#
# DPO 的创新：
#   通过数学推导，把"最大化人类偏好奖励"转化为一个
#   纯监督的对比学习损失，不再需要单独的奖励模型和 PPO：
#
#     Loss_DPO = -E[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x))
#                         - β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
#
#   其中：
#     π_θ     = 当前训练的策略模型（初始化为 SFT 模型）
#     π_ref   = 冻结的参考模型（SFT 模型的副本）
#     y_w     = 人类偏好的优选回答 (chosen)
#     y_l     = 人类不喜欢的劣选回答 (rejected)
#     β       = KL 惩罚系数，控制偏离参考模型的程度（通常 0.1~0.5）
#
# 直觉理解：
#   - 提高 y_w 相对参考模型的生成概率
#   - 降低 y_l 相对参考模型的生成概率
#   - β 越大：越保守，越接近 SFT 模型
#   - β 越小：越大胆，更激进地优化偏好
#
# 数据格式（偏好对）：
#   每条样本包含：
#     "prompt"   - 输入问题
#     "chosen"   - 人类偏好的好回答
#     "rejected" - 人类不喜欢的差回答
#
# 参考：DPO 原论文 https://arxiv.org/abs/2305.18290
# ============================================================

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """解析 DPO 训练参数。"""
    parser = argparse.ArgumentParser(description="使用 DPO 对齐 Qwen 模型。")
    parser.add_argument("--config", default="configs/rl_dpo.yaml")
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置，不加载模型。")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """DPO 训练主入口。"""
    args = parse_args()
    cfg = load_yaml(args.config)

    model_name = cfg["model"]["model_name_or_path"]
    train_file = cfg["data"]["train_file"]
    output_dir = cfg["training"]["output_dir"]
    beta = cfg["training"].get("beta", 0.1)

    print(f"基座模型:     {model_name}")
    print(f"偏好数据:     {train_file}")
    print(f"输出目录:     {output_dir}")
    print(f"DPO beta:     {beta}  (KL 惩罚系数，越大越保守)")

    if args.dry_run:
        print("Dry run 完成：配置合法。")
        return

    import json

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 策略模型和参考模型均从 SFT checkpoint 初始化。
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype="auto"
    )

    training_cfg = cfg["training"]
    lora_cfg = cfg.get("lora", {})
    peft_config = None
    if lora_cfg:
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
            task_type="CAUSAL_LM",
        )

    def _load_preference_jsonl(path: str) -> list[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    rows = _load_preference_jsonl(train_file)
    dataset = Dataset.from_list(rows)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=float(beta),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        learning_rate=float(training_cfg["learning_rate"]),
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        report_to="none",
        max_length=int(cfg["data"].get("max_seq_length", 512)),
        max_prompt_length=int(cfg["data"].get("max_prompt_length", 256)),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None 时 DPOTrainer 自动克隆参考模型
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"DPO 对齐模型已保存至: {output_dir}")


if __name__ == "__main__":
    main()

