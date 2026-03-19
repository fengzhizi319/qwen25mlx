from __future__ import annotations


# ============================================================
# 【原理说明】奖励模型 (Reward Model)
# ============================================================
#
# 强化学习对齐（RLHF）的核心组件之一是奖励模型（RM）。
# 奖励模型的作用：
#   给定一段对话 (prompt, response)，预测人类偏好得分，
#   供后续 PPO/DPO 训练使用。
#
# 训练方式（Bradley-Terry 偏好模型）：
#   给定同一 prompt 的"优选回答 y_w"和"劣选回答 y_l"：
#
#     Loss = -log σ( r(x, y_w) - r(x, y_l) )
#
#   σ 为 sigmoid，r(x, y) 是奖励模型对回答 y 的打分。
#   目标：让好回答得分始终高于差回答。
#
# 本脚本：
#   - 定义奖励模型的输入/输出接口
#   - 实现基于 sequence 最后一个 token 的标量打分
#   - 提供 reward score 的 dry-run 验证入口
# ============================================================

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """解析奖励模型训练参数。"""
    parser = argparse.ArgumentParser(description="训练奖励模型（RM）。")
    parser.add_argument("--config", default="configs/rl_reward_model.yaml")
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置，不加载模型。")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_reward_score(logits: "list[float]") -> float:
    """
    从序列末尾 logit 中取标量奖励分。

    原理：
      奖励模型在 CausalLM 头之上接一个线性层，输出维度为 1。
      取序列最后一个有效 token 的输出值作为该回答的整体得分。
    """
    # 取最后一个 token 的 logit 作为标量奖励。
    return float(logits[-1]) if logits else 0.0


def main() -> None:
    """奖励模型训练主入口（dry-run 验证配置）。"""
    args = parse_args()
    cfg = load_yaml(args.config)

    model_name = cfg["model"]["model_name_or_path"]
    train_file = cfg["data"]["train_file"]
    output_dir = cfg["training"]["output_dir"]

    print(f"奖励模型基座:  {model_name}")
    print(f"偏好数据文件:  {train_file}")
    print(f"输出目录:      {output_dir}")

    if args.dry_run:
        print("Dry run 完成：配置合法。")
        return

    # 实际训练需安装 trl >= 0.10，使用 RewardTrainer。
    from datasets import Dataset
    from trl import RewardConfig, RewardTrainer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # 奖励模型需要 pad_token 对批次做 padding。
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,   # 输出一个标量奖励分
        trust_remote_code=True,
        torch_dtype="auto",
    )

    import json

    def _load_preference_jsonl(path: str) -> list[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _tokenize_pair(row: dict) -> dict:
        chosen_enc = tokenizer(
            row["chosen"], truncation=True, max_length=cfg["data"]["max_seq_length"]
        )
        rejected_enc = tokenizer(
            row["rejected"], truncation=True, max_length=cfg["data"]["max_seq_length"]
        )
        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    rows = _load_preference_jsonl(train_file)
    dataset = Dataset.from_list([_tokenize_pair(r) for r in rows])

    training_cfg = cfg["training"]
    reward_config = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        learning_rate=float(training_cfg["learning_rate"]),
        logging_steps=int(training_cfg.get("logging_steps", 5)),
        report_to="none",
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"奖励模型已保存至: {output_dir}")


if __name__ == "__main__":
    main()

