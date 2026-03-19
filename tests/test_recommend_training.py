from __future__ import annotations

from src.common.recommend_training import build_config_snippet, recommend_from_memory_gb


# 教学注释块
# 测试目的: 验证在同等内存下，MLX 的建议样本规模通常不小于 TRL。
# 关键断言: max_samples(mlx) >= max_samples(trl)。
def test_recommendation_mlx_not_smaller_than_trl_for_same_memory() -> None:
    trl = recommend_from_memory_gb(memory_gb=16, framework="trl", profile="balanced")
    mlx = recommend_from_memory_gb(memory_gb=16, framework="mlx", profile="balanced")
    assert mlx["max_samples"] >= trl["max_samples"]


# 教学注释块
# 测试目的: 验证不同 profile 会改变建议规模。
# 关键断言: aggressive > balanced > safe。
def test_profile_scale_changes_max_samples() -> None:
    safe = recommend_from_memory_gb(memory_gb=32, framework="trl", profile="safe")
    balanced = recommend_from_memory_gb(memory_gb=32, framework="trl", profile="balanced")
    aggressive = recommend_from_memory_gb(memory_gb=32, framework="trl", profile="aggressive")
    assert safe["max_samples"] < balanced["max_samples"] < aggressive["max_samples"]


# 教学注释块
# 测试目的: 验证 TRL 配置片段字段名正确。
# 关键断言: 包含 per_device_train_batch_size 和 gradient_accumulation_steps。
def test_build_config_snippet_for_trl() -> None:
    rec = recommend_from_memory_gb(memory_gb=24, framework="trl", profile="balanced")
    snippet = build_config_snippet(rec)
    assert "dataset_download" in snippet
    assert "training" in snippet
    assert "per_device_train_batch_size" in snippet["training"]
    assert "gradient_accumulation_steps" in snippet["training"]


# 教学注释块
# 测试目的: 验证 MLX 配置片段字段名正确。
# 关键断言: 训练字段使用 batch_size。
def test_build_config_snippet_for_mlx() -> None:
    rec = recommend_from_memory_gb(memory_gb=24, framework="mlx", profile="balanced")
    snippet = build_config_snippet(rec)
    assert "dataset_download" in snippet
    assert "training" in snippet
    assert "batch_size" in snippet["training"]

