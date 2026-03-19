from __future__ import annotations

from pathlib import Path

from src.trl_path.train_sft import find_latest_checkpoint, resolve_report_to, resolve_resume_checkpoint


# 教学注释块
# 测试目的: 验证 checkpoint 自动选择逻辑。
# 关键断言: 选择步数最大的 checkpoint 目录。
def test_find_latest_checkpoint_returns_highest_step(tmp_path: Path) -> None:
    """学习测试：应选择步数最大的 checkpoint。"""
    out_dir = tmp_path / "out"
    (out_dir / "checkpoint-10").mkdir(parents=True)
    (out_dir / "checkpoint-40").mkdir(parents=True)
    (out_dir / "checkpoint-20").mkdir(parents=True)

    assert find_latest_checkpoint(out_dir) == str(out_dir / "checkpoint-40")


# 教学注释块
# 测试目的: 验证多日志后端字符串解析。
# 关键断言: 逗号分隔可转成列表。
def test_resolve_report_to_from_comma_text() -> None:
    """支持从配置里解析多后端上报。"""
    cfg = {"experiment": {"report_to": "tensorboard,wandb"}}
    assert resolve_report_to(None, cfg) == ["tensorboard", "wandb"]


# 教学注释块
# 测试目的: 验证 auto 续训会扫描最新断点。
# 关键断言: 返回最高步数 checkpoint 路径。
def test_resolve_resume_checkpoint_auto(tmp_path: Path) -> None:
    """auto 模式应自动定位最新断点。"""
    out_dir = tmp_path / "train_out"
    (out_dir / "checkpoint-15").mkdir(parents=True)

    cfg = {"training": {"output_dir": str(out_dir), "resume_from_checkpoint": "auto"}}
    assert resolve_resume_checkpoint(None, cfg) == str(out_dir / "checkpoint-15")


# 教学注释块
# 测试目的: 验证命令行参数优先级。
# 关键断言: CLI 的 checkpoint 覆盖配置文件值。
def test_resolve_resume_checkpoint_cli_overrides_cfg(tmp_path: Path) -> None:
    """CLI 传参优先级高于配置文件。"""
    out_dir = tmp_path / "train_out"
    cfg = {"training": {"output_dir": str(out_dir), "resume_from_checkpoint": "none"}}
    assert resolve_resume_checkpoint("/tmp/custom_ckpt", cfg) == "/tmp/custom_ckpt"


# 教学注释块
# 测试目的: 验证默认日志后端行为。
# 关键断言: 未配置时应返回 none。
def test_resolve_report_to_none_by_default() -> None:
    """默认应关闭外部上报，避免学习阶段额外依赖。"""
    assert resolve_report_to(None, {}) == "none"


# 教学注释块
# 测试目的: 验证 none 模式不进行断点续训。
# 关键断言: 返回值为 None。
def test_resolve_resume_checkpoint_none() -> None:
    """none 模式不续训。"""
    cfg = {"training": {"output_dir": "outputs/trl_lora_adapter", "resume_from_checkpoint": "none"}}
    assert resolve_resume_checkpoint(None, cfg) is None


