from __future__ import annotations

from pathlib import Path

from src.common.data import build_text_records, read_jsonl


def main() -> None:
    """演示数据加载与 prompt 拼接效果，作为学习入口。"""
    train_file = Path("data/minimal/train.jsonl")
    rows = read_jsonl(train_file)
    records = build_text_records(rows)

    print(f"Loaded samples: {len(rows)}")
    print("\nFirst formatted sample:\n")
    print(records[0]["text"])


if __name__ == "__main__":
    main()

