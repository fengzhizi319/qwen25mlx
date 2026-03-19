from __future__ import annotations

# ============================================================
# 【测试模块说明】强化学习对齐 (RL Alignment) 测试套件
# ============================================================
#
# 本模块覆盖 RLHF 三大核心组件的逻辑测试（不需要真实模型权重）：
#   1. 奖励模型（Reward Model）         → train_reward_model.py
#   2. DPO 直接偏好优化                  → train_dpo.py
#   3. PPO 近端策略优化                  → train_ppo.py
#
# 测试策略：
#   - 所有需要实际模型的测试走 dry-run，仅校验配置文件合法性。
#   - 纯逻辑函数（数据解析、奖励分计算）直接测试，无需 GPU。
#   - 数据格式测试验证 preference_train.jsonl 字段完整性。
# ============================================================

import json
from pathlib import Path

import pytest
import yaml


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ────────────────────────────────────────────────────────────
# 1. 数据格式测试
# ────────────────────────────────────────────────────────────

# 教学注释块
# 测试目的: 验证偏好数据集包含 prompt/chosen/rejected 三元组。
# 关键断言: 每条样本三个字段都存在且非空。
def test_preference_data_has_required_fields() -> None:
    """偏好数据集格式校验：必须包含 prompt/chosen/rejected。"""
    rows = load_jsonl("data/rl/preference_train.jsonl")
    assert len(rows) > 0, "偏好数据集不能为空。"
    for i, row in enumerate(rows):
        for key in ("prompt", "chosen", "rejected"):
            assert key in row, f"第 {i+1} 条样本缺少字段: {key}"
            assert str(row[key]).strip(), f"第 {i+1} 条样本字段 {key} 不能为空。"


# 教学注释块
# 测试目的: 验证优选回答质量通常优于劣选（长度启发式检验）。
# 关键断言: chosen 长度 >= rejected 长度（至少一半样本满足）。
def test_preference_chosen_generally_longer_than_rejected() -> None:
    """偏好数据集质量检查：chosen 通常比 rejected 信息量更丰富。"""
    rows = load_jsonl("data/rl/preference_train.jsonl")
    count_longer = sum(1 for r in rows if len(r["chosen"]) >= len(r["rejected"]))
    assert count_longer >= len(rows) // 2, "chosen 质量低于预期，请检查数据标注。"


# 教学注释块
# 测试目的: 验证 PPO 用的纯 prompt 数据集格式正确。
# 关键断言: 每条样本只需包含 prompt 字段。
def test_prompts_data_has_prompt_field() -> None:
    """PPO 训练所需 prompt 数据集格式验证。"""
    rows = load_jsonl("data/rl/prompts_train.jsonl")
    assert len(rows) > 0
    for i, row in enumerate(rows):
        assert "prompt" in row, f"第 {i+1} 条样本缺少 prompt 字段。"


# ────────────────────────────────────────────────────────────
# 2. 配置文件测试
# ────────────────────────────────────────────────────────────

# 教学注释块
# 测试目的: 验证奖励模型 YAML 包含必要训练配置项。
# 关键断言: model/data/training 三个顶级 key 都存在。
def test_reward_model_config_has_required_sections() -> None:
    """奖励模型配置文件结构校验。"""
    cfg = load_yaml("configs/rl_reward_model.yaml")
    for section in ("model", "data", "training"):
        assert section in cfg, f"奖励模型配置缺少 [{section}] 段。"
    assert "model_name_or_path" in cfg["model"]
    assert "train_file" in cfg["data"]
    assert "output_dir" in cfg["training"]


# 教学注释块
# 测试目的: 验证 DPO 配置中 beta 值在合理区间。
# 关键断言: 0 < beta < 2（过大会导致训练退化为 SFT）。
def test_dpo_config_beta_in_valid_range() -> None:
    """DPO 配置 beta 系数合理性校验。"""
    cfg = load_yaml("configs/rl_dpo.yaml")
    beta = float(cfg["training"]["beta"])
    assert 0 < beta < 2, f"DPO beta={beta} 超出合理区间 (0, 2)。"


# 教学注释块
# 测试目的: 验证 PPO 配置包含必要的模型路径（actor + reward）。
# 关键断言: model_name_or_path 和 reward_model_path 均配置。
def test_ppo_config_has_both_model_paths() -> None:
    """PPO 配置需同时指定策略模型和奖励模型。"""
    cfg = load_yaml("configs/rl_ppo.yaml")
    assert "model_name_or_path" in cfg["model"], "缺少 actor 模型路径。"
    assert "reward_model_path" in cfg["model"], "缺少奖励模型路径。"


# 教学注释块
# 测试目的: 验证三个 RL 配置文件使用相同的基座模型。
# 关键断言: 三个配置的 model_name_or_path 一致，保证实验对比公平。
def test_all_rl_configs_use_same_base_model() -> None:
    """三个 RL 配置文件应使用同一基座模型，保证可比性。"""
    rm_cfg = load_yaml("configs/rl_reward_model.yaml")
    dpo_cfg = load_yaml("configs/rl_dpo.yaml")
    ppo_cfg = load_yaml("configs/rl_ppo.yaml")
    base_rm = rm_cfg["model"]["model_name_or_path"]
    base_dpo = dpo_cfg["model"]["model_name_or_path"]
    base_ppo = ppo_cfg["model"]["model_name_or_path"]
    assert base_rm == base_dpo == base_ppo, (
        f"基座模型不一致: RM={base_rm}, DPO={base_dpo}, PPO={base_ppo}"
    )


# ────────────────────────────────────────────────────────────
# 3. 奖励模型核心逻辑测试
# ────────────────────────────────────────────────────────────

# 教学注释块
# 测试目的: 验证 compute_reward_score 从 logit 列表正确取末位标量。
# 关键断言: 返回最后一个元素的 float 值。
def test_compute_reward_score_returns_last_logit() -> None:
    """奖励分计算：取序列末尾 token 的 logit 作为标量得分。"""
    from src.rl_path.train_reward_model import compute_reward_score

    assert compute_reward_score([0.1, 0.5, 0.9]) == pytest.approx(0.9)
    assert compute_reward_score([1.2]) == pytest.approx(1.2)


# 教学注释块
# 测试目的: 验证空 logit 列表不崩溃，返回 0。
# 关键断言: compute_reward_score([]) == 0.0。
def test_compute_reward_score_empty_returns_zero() -> None:
    """奖励分计算对空输入应安全返回 0，不抛出异常。"""
    from src.rl_path.train_reward_model import compute_reward_score

    assert compute_reward_score([]) == 0.0


# ────────────────────────────────────────────────────────────
# 4. Dry-run 集成测试
# ────────────────────────────────────────────────────────────

# 教学注释块
# 测试目的: 奖励模型 dry-run 应读取配置并打印关键路径，不报错。
# 关键断言: 函数正常返回，无异常抛出。
def test_reward_model_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """奖励模型 dry-run：仅校验配置合法性，不加载模型权重。"""
    monkeypatch.setattr(
        "sys.argv", ["train_reward_model.py", "--config", "configs/rl_reward_model.yaml", "--dry-run"]
    )
    from src.rl_path.train_reward_model import main
    main()  # 不抛出异常即通过


# 教学注释块
# 测试目的: DPO dry-run 应读取配置并打印 beta 系数，不报错。
# 关键断言: 函数正常返回，无异常抛出。
def test_dpo_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """DPO dry-run：校验配置，不触发模型下载和训练。"""
    monkeypatch.setattr(
        "sys.argv", ["train_dpo.py", "--config", "configs/rl_dpo.yaml", "--dry-run"]
    )
    from src.rl_path.train_dpo import main
    main()


# 教学注释块
# 测试目的: PPO dry-run 应读取配置并打印 actor/reward 模型路径，不报错。
# 关键断言: 函数正常返回，无异常抛出。
def test_ppo_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """PPO dry-run：校验配置，不触发模型加载和在线采样。"""
    monkeypatch.setattr(
        "sys.argv", ["train_ppo.py", "--config", "configs/rl_ppo.yaml", "--dry-run"]
    )
    from src.rl_path.train_ppo import main
    main()


# ────────────────────────────────────────────────────────────
# 5. 算法对比知识测试
# ────────────────────────────────────────────────────────────

# 教学注释块
# 测试目的: 以代码形式固化 DPO beta 与训练保守程度的关系。
# 关键断言: beta 越大越接近参考模型（不是越激进）。
def test_dpo_beta_interpretation() -> None:
    """DPO beta 值的语义验证：越大越保守，越接近 SFT 参考模型。"""
    # beta 控制 KL 惩罚强度：
    #   beta -> 0: 完全优化偏好，可能偏离参考模型很远（reward hacking）
    #   beta -> ∞: 完全复现参考模型，无法优化偏好
    # 推荐值 0.1 ~ 0.5 是工程实践中的平衡点。
    cfg = load_yaml("configs/rl_dpo.yaml")
    beta = float(cfg["training"]["beta"])
    assert 0.05 <= beta <= 0.5, f"beta={beta} 超出推荐范围 [0.05, 0.5]。"


# 教学注释块
# 测试目的: 验证偏好数据的 chosen/rejected 确实不同。
# 关键断言: 任何样本中 chosen != rejected。
def test_preference_chosen_differs_from_rejected() -> None:
    """基本数据质量保证：chosen 和 rejected 不能是相同文本。"""
    rows = load_jsonl("data/rl/preference_train.jsonl")
    for i, row in enumerate(rows):
        assert row["chosen"] != row["rejected"], (
            f"第 {i+1} 条样本 chosen 与 rejected 相同，数据标注有误。"
        )

