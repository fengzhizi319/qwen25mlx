from __future__ import annotations


# ============================================================
# 【原理说明】SFT 数据流水线 (Supervised Fine-Tuning Pipeline)
# ============================================================
#
# 监督微调（SFT）的本质是让模型在"输入序列"条件下最大化
# "输出序列"的对数似然（log-likelihood）：
#
#   Loss = -Σ log P(token_i | token_1...token_{i-1}, instruction)
#
# 数据流水线的目标就是把原始三元组转成这条损失函数可用的格式：
#
#   原始               →  JSONL 三元组         →  text 字段
#   {"instruction":…      read_jsonl()            build_text_records()
#    "input":…            校验字段完整性          拼成训练文本
#    "output":…}          跳过空行                → {"text": "### Instruction:…"}
#
# 关键设计决定：
#   1. text 字段同时包含 instruction 和 response，模型在训练时对
#      整段 text 做语言建模（不只是 response 部分），这是 Causal LM
#      监督学习的标准做法。
#   2. JSONL 每行一条样本，易于流式读取，不需要把整个数据集载入内存。
# ============================================================

import json
from pathlib import Path
from typing import Iterable

from src.common.prompting import format_instruction_sample


REQUIRED_KEYS = ("instruction", "input", "output")


def read_jsonl(path: str | Path) -> list[dict]:
    """读取并校验 JSONL，确保每条样本包含训练所需字段。"""
    file_path = Path(path)
    rows: list[dict] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            # 跳过空行，方便手工编辑数据集时保留可读性。
            if not stripped:
                continue
            obj = json.loads(stripped)
            missing = [k for k in REQUIRED_KEYS if k not in obj]
            if missing:
                raise ValueError(
                    f"Missing keys {missing} in {file_path} at line {line_number}."
                )
            rows.append(obj)
    return rows


def build_text_records(rows: Iterable[dict]) -> list[dict]:
    """把 instruction/input/output 转为 SFT 常用的 `text` 字段。"""
    records: list[dict] = []
    for row in rows:
        records.append(
            {
                # 将结构化样本拼成可直接喂给语言模型的监督文本。
                "text": format_instruction_sample(
                    instruction=row["instruction"],
                    input_text=row.get("input", ""),
                    output_text=row["output"],
                )
            }
        )
    return records

