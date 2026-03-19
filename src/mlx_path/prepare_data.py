from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.data import build_text_records, read_jsonl


def parse_args() -> argparse.Namespace:
    """解析 MLX 数据转换参数。"""
    parser = argparse.ArgumentParser(description="Prepare MLX jsonl data with a `text` field.")
    parser.add_argument("--train-file", default="data/minimal/train.jsonl")
    parser.add_argument("--eval-file", default="data/minimal/eval.jsonl")
    parser.add_argument("--output-dir", default="data/mlx")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """将样本列表写入 JSONL 文件。"""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """把项目通用数据转换为 MLX LoRA 训练可直接读取的格式。"""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_text_records(read_jsonl(args.train_file))
    eval_rows = build_text_records(read_jsonl(args.eval_file))

    train_out = out_dir / "train.jsonl"
    valid_out = out_dir / "valid.jsonl"

    write_jsonl(train_out, train_rows)
    write_jsonl(valid_out, eval_rows)

    print(f"Saved MLX train file: {train_out}")
    print(f"Saved MLX valid file: {valid_out}")


if __name__ == "__main__":
    main()

