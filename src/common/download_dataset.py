from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATASET = "yahma/alpaca-cleaned"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，用于从 Hugging Face 下载并转换训练集。"""
    parser = argparse.ArgumentParser(description="下载并转换指令微调数据集为本项目 JSONL 格式。")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face datasets 数据集名称")
    parser.add_argument("--split", default="train", help="要下载的数据集切分")
    parser.add_argument("--max-samples", type=int, default=2000, help="最多保留样本数，便于学习阶段快速实验")
    parser.add_argument("--eval-ratio", type=float, default=0.02, help="划分验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="切分随机种子")
    parser.add_argument("--instruction-col", default="instruction")
    parser.add_argument("--input-col", default="input")
    parser.add_argument("--output-col", default="output")
    parser.add_argument("--out-train", default="data/downloaded/train.jsonl")
    parser.add_argument("--out-eval", default="data/downloaded/eval.jsonl")
    return parser.parse_args()


def map_row(row: dict, instruction_col: str, input_col: str, output_col: str) -> dict:
    """将任意列名映射到项目统一的 instruction/input/output 三元组。"""
    return {
        "instruction": str(row.get(instruction_col, "")).strip(),
        "input": str(row.get(input_col, "")).strip(),
        "output": str(row.get(output_col, "")).strip(),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """按 JSONL 写出，确保中文可读。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if not (0.0 < args.eval_ratio < 1.0):
        raise ValueError("--eval-ratio 必须在 0 和 1 之间。")
    if args.max_samples <= 1:
        raise ValueError("--max-samples 必须大于 1。")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "需要安装 datasets。请先执行: pip install -r requirements-trl.txt"
        ) from exc

    # 先拉取数据，再做截断和切分，保证学习阶段可快速复现实验。
    ds = load_dataset(args.dataset, split=args.split)
    if len(ds) == 0:
        raise ValueError("下载的数据集为空，请检查数据集名称和 split。")

    selected = ds.select(range(min(args.max_samples, len(ds))))
    split_ds = selected.train_test_split(test_size=args.eval_ratio, seed=args.seed)

    train_rows = [
        map_row(row, args.instruction_col, args.input_col, args.output_col)
        for row in split_ds["train"]
    ]
    eval_rows = [
        map_row(row, args.instruction_col, args.input_col, args.output_col)
        for row in split_ds["test"]
    ]

    out_train = Path(args.out_train)
    out_eval = Path(args.out_eval)
    write_jsonl(out_train, train_rows)
    write_jsonl(out_eval, eval_rows)

    print(f"已保存训练集: {out_train} (样本数: {len(train_rows)})")
    print(f"已保存验证集: {out_eval} (样本数: {len(eval_rows)})")


if __name__ == "__main__":
    main()

