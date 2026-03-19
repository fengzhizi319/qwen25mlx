from __future__ import annotations

from argparse import Namespace

from src.mlx_path.infer import build_command as build_infer_command
from src.mlx_path.train_lora import build_command as build_train_command


# 教学注释块
# 测试目的: 验证 MLX 训练命令包含核心参数开关。
# 关键断言: model/data/adapter-path 三个关键 flag 存在。
def test_build_train_command_contains_required_flags() -> None:
    """学习测试：MLX 训练命令应包含关键超参数。"""
    cfg = {
        "model": {"model_path": "Qwen/Qwen2.5-0.5B"},
        "training": {
            "data_dir": "data/mlx",
            "adapter_path": "outputs/mlx_adapters",
            "iters": 10,
            "batch_size": 1,
            "learning_rate": 1e-5,
            "steps_per_eval": 5,
            "steps_per_report": 2,
            "save_every": 10,
        },
    }

    cmd = build_train_command(cfg)
    assert "--model" in cmd
    assert "--data" in cmd
    assert "--adapter-path" in cmd


# 教学注释块
# 测试目的: 验证推理命令会透传 adapter 路径。
# 关键断言: --adapter-path 和其值同时存在。
def test_build_infer_command_with_adapter() -> None:
    """学习测试：设置 adapter 路径时应透传到推理命令。"""
    args = Namespace(
        model_path="Qwen/Qwen2.5-0.5B",
        adapter_path="outputs/mlx_adapters",
        prompt="hello",
        max_tokens=16,
        temperature=0.7,
    )

    cmd = build_infer_command(args)
    assert "--adapter-path" in cmd
    assert "outputs/mlx_adapters" in cmd

