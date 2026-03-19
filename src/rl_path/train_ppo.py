from __future__ import annotations


# ============================================================
# 【原理说明】PPO 近端策略优化 (Proximal Policy Optimization)
# ============================================================
#
# PPO 是 RLHF 的经典在线对齐算法，OpenAI InstructGPT 采用的路径。
#
# 核心目标：
#   最大化奖励 R 的同时，用 KL 散度惩罚防止策略偏离太远：
#
#     J(θ) = E[R(x,y)] - β·KL( π_θ(y|x) || π_ref(y|x) )
#
# 与 DPO 对比：
#   ┌───────────────┬──────────────────┬─────────────────┐
#   │               │ PPO              │ DPO             │
#   ├───────────────┼──────────────────┼─────────────────┤
#   │ 训练复杂度    │ 高（4个模型）    │ 低（2个模型）    │
#   │ 在线/离线     │ 在线（需采样）   │ 离线（静态数据） │
#   │ 奖励信号      │ 奖励模型打分     │ 偏好对隐式奖励   │
#   │ 计算资源      │ 高               │ 低               │
#   │ 效果上限      │ 更高（理论上）   │ 实践中相近       │
#   └───────────────┴──────────────────┴─────────────────┘
#
# PPO 四个模型：
#   1. actor_model     = 当前策略（持续更新）
#   2. ref_model       = 参考策略（冻结，SFT 初始化）
#   3. reward_model    = 奖励模型（冻结，RM 训练得到）
#   4. critic_model    = 价值网络（估计状态价值）
#
# 本脚本封装 TRL PPOTrainer，负责：
#   - 加载 actor / ref / reward 模型
#   - 生成 response 并用 reward_model 打分
#   - 运行 PPO 更新 actor 权重
# ============================================================

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """解析 PPO 训练参数。"""
    parser = argparse.ArgumentParser(description="使用 PPO 对齐 Qwen 模型（RLHF）。")
    parser.add_argument("--config", default="configs/rl_ppo.yaml")
    parser.add_argument("--dry-run", action="store_true", help="仅校验配置，不加载模型。")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """PPO 训练主入口。"""
    args = parse_args()
    cfg = load_yaml(args.config)

    model_name = cfg["model"]["model_name_or_path"]
    reward_model_path = cfg["model"]["reward_model_path"]
    train_file = cfg["data"]["train_file"]
    output_dir = cfg["training"]["output_dir"]

    print(f"Actor 基座:        {model_name}")
    print(f"奖励模型路径:      {reward_model_path}")
    print(f"训练数据:          {train_file}")
    print(f"输出目录:          {output_dir}")

    if args.dry_run:
        print("Dry run 完成：配置合法。")
        return

    import json

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from trl import PPOConfig, PPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_cfg = cfg["training"]
    lora_cfg = cfg.get("lora", {})

    # Actor 模型：从 SFT checkpoint 初始化，持续被 PPO 更新。
    actor_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype="auto"
    )

    peft_config = None
    if lora_cfg:
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
            task_type="CAUSAL_LM",
        )

    # 奖励模型：冻结，用于对生成的 response 打分。
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, num_labels=1, trust_remote_code=True, torch_dtype="auto"
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)

    def _load_prompts(path: str) -> list[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    prompts = _load_prompts(train_file)
    dataset = Dataset.from_list(prompts)

    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=float(training_cfg["learning_rate"]),
        batch_size=int(training_cfg.get("batch_size", 1)),
        mini_batch_size=int(training_cfg.get("mini_batch_size", 1)),
        ppo_epochs=int(training_cfg.get("ppo_epochs", 4)),
        log_with=None,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=actor_model,
        ref_model=None,   # None 时 PPOTrainer 自动克隆参考模型
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=None,
        peft_config=peft_config,
    )

    # PPO 在线训练循环：生成 → 打分 → 更新。
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    reward_model = reward_model.to(device)

    for _batch in trainer.dataloader:
        query_tensors = _batch["input_ids"]

        # 策略模型生成 response。
        response_tensors = trainer.generate(query_tensors, max_new_tokens=64)
        decoded = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # 奖励模型对 response 打分。
        reward_inputs = reward_tokenizer(decoded, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            rewards_raw = reward_model(**reward_inputs).logits.squeeze(-1)
        rewards = [r for r in rewards_raw]

        # PPO 更新 actor。
        trainer.step(query_tensors, response_tensors, rewards)

    trainer.save_pretrained(output_dir)
    print(f"PPO 对齐模型已保存至: {output_dir}")


if __name__ == "__main__":
    main()

