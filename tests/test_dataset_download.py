from __future__ import annotations

import json
import os
from pathlib import Path
from time import strftime

import pytest

from src.common.download_dataset import main as download_main
from src.common.download_dataset import map_row, write_jsonl


def backup_existing_dataset_files(dataset_dir: Path) -> None:
    """将已有 train/eval 文件移动到 datasets/baks，避免覆盖历史结果。"""
    backup_dir = dataset_dir / "baks"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ("train.jsonl", "eval.jsonl"):
        src = dataset_dir / file_name
        if not src.exists():
            continue

        # 使用时间戳避免多次测试时备份重名。
        timestamp = strftime("%Y%m%d_%H%M%S")
        dst = backup_dir / f"{src.stem}_{timestamp}{src.suffix}"
        src.replace(dst)


# 教学注释块
# 测试目的: 验证自定义列名能映射到统一三元组格式。
# 关键断言: 输出只包含 instruction/input/output。
def test_map_row_converts_columns() -> None:
    """学习测试：支持将不同列名映射到统一训练格式。"""
    row = {"inst": "Do", "ctx": "in", "ans": "out"}
    mapped = map_row(row, "inst", "ctx", "ans")
    assert mapped == {"instruction": "Do", "input": "in", "output": "out"}


# 教学注释块
# 测试目的: 验证 JSONL 写出时保留中文字符。
# 关键断言: 读取后中文内容与原值一致。
def test_write_jsonl_keeps_utf8(tmp_path: Path) -> None:
    """学习测试：写出的 JSONL 应保留中文字符。"""
    out = tmp_path / "train.jsonl"
    write_jsonl(out, [{"instruction": "翻译", "input": "hello", "output": "你好"}])

    line = out.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["output"] == "你好"


# 教学注释块
# 测试目的: 验证非法参数会在下载前被拦截。
# 关键断言: eval-ratio 越界时抛出 ValueError。
def test_main_raises_on_invalid_eval_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """参数校验在下载前执行，非法比例会直接报错。"""
    monkeypatch.setattr(
        "sys.argv",
        [
            "download_dataset.py",
            "--eval-ratio",
            "1.2",
        ],
    )

    with pytest.raises(ValueError):
        download_main()


# 教学注释块
# 测试目的: 将“下载并切分训练集”作为一个集成测试入口。
# 关键断言: 生成 train/eval 文件且样本总数不超过 max-samples。
def test_download_and_split_dataset_integration(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUN_DATASET_DOWNLOAD_TEST", "1")
    """集成测试：可选执行真实下载与切分流程。"""
    if os.environ.get("RUN_DATASET_DOWNLOAD_TEST") != "1":
        pytest.skip("默认跳过真实下载测试；设置 RUN_DATASET_DOWNLOAD_TEST=1 后执行。")

    pytest.importorskip("datasets")
    dataset_dir = Path("datasets")
    out_train = dataset_dir / "train.jsonl"
    out_eval = dataset_dir / "eval.jsonl"
    # 可通过环境变量调大真实下载 UT 的样本量。
    max_samples = int(os.environ.get("RUN_REAL_DOWNLOAD_MAX_SAMPLES", "30000"))
    if max_samples <= 1:
        raise ValueError("RUN_REAL_DOWNLOAD_MAX_SAMPLES 必须大于 1。")

    # 测试前先备份旧文件，避免历史数据影响本次观察。
    backup_existing_dataset_files(dataset_dir)

    monkeypatch.setattr(
        "sys.argv",
        [
            "download_dataset.py",
            "--dataset",
            "yahma/alpaca-cleaned",
            "--split",
            "train",
            "--max-samples",
            str(max_samples),
            "--eval-ratio",
            "0.2",
            "--out-train",
            str(out_train),
            "--out-eval",
            str(out_eval),
        ],
    )

    download_main()

    assert out_train.exists()
    assert out_eval.exists()

    train_lines = out_train.read_text(encoding="utf-8").strip().splitlines()
    eval_lines = out_eval.read_text(encoding="utf-8").strip().splitlines()
    assert len(train_lines) > 0
    assert len(eval_lines) > 0
    assert len(train_lines) + len(eval_lines) <= max_samples

    sample = json.loads(train_lines[0])
    assert set(sample.keys()) == {"instruction", "input", "output"}

    # 输出目录改为项目根目录下的 datasets，便于重复实验复用。
    assert out_train.parent == dataset_dir


