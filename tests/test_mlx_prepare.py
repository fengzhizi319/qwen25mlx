from __future__ import annotations

import json
from pathlib import Path

from src.mlx_path.prepare_data import main as prepare_main


# 教学注释块
# 测试目的: 验证 MLX 数据准备会生成 train/valid 两个文件。
# 关键断言: 文件存在且样本结构包含 text 字段。
def test_prepare_data_writes_train_and_valid(tmp_path: Path, monkeypatch) -> None:
    """学习测试：MLX 数据准备会生成 train/valid 且包含 text 字段。"""
    out_dir = tmp_path / "mlx"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--train-file",
            "data/minimal/train.jsonl",
            "--eval-file",
            "data/minimal/eval.jsonl",
            "--output-dir",
            str(out_dir),
        ],
    )

    prepare_main()

    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "valid.jsonl").exists()

    first_line = (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    assert "text" in record

