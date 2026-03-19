from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.data import build_text_records, read_jsonl


# 教学注释块
# 测试目的: 验证最小样本能被读取并转成训练 text。
# 关键断言: 记录数量一致且模板字段存在。
def test_read_jsonl_and_build_text_records() -> None:
    """基础测试：最小数据集可以读取并转换为 text 字段。"""
    rows = read_jsonl(Path("data/minimal/train.jsonl"))
    assert len(rows) >= 5

    records = build_text_records(rows)
    assert len(records) == len(rows)
    assert "### Instruction:" in records[0]["text"]
    assert "### Response:" in records[0]["text"]


# 教学注释块
# 测试目的: 验证 JSONL 中空行会被安全忽略。
# 关键断言: 两个换行只计入一条样本。
def test_read_jsonl_skips_empty_lines(tmp_path: Path) -> None:
    """学习测试：验证空行会被忽略，避免手工编辑数据时报错。"""
    file_path = tmp_path / "train.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"instruction": "A", "input": "", "output": "B"}) + "\n\n")

    rows = read_jsonl(file_path)
    assert len(rows) == 1


# 教学注释块
# 测试目的: 验证缺字段样本会触发显式异常。
# 关键断言: 抛出 ValueError，帮助定位坏数据。
def test_read_jsonl_raises_when_missing_key(tmp_path: Path) -> None:
    """学习测试：缺字段数据会被明确拦截，便于定位坏样本。"""
    file_path = tmp_path / "bad.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"instruction": "A", "output": "B"}) + "\n")

    with pytest.raises(ValueError):
        read_jsonl(file_path)


