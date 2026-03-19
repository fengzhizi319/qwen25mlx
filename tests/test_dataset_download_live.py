from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.common.download_dataset import main as download_main


# 教学注释块
# 测试目的: 执行一次真实网络下载 + 切分，验证端到端可用性。
# 前置条件: 设置 RUN_REAL_DOWNLOAD_UT=1，并保证网络可访问 Hugging Face。
# 关键断言: 产出 train/eval 文件，且样本总数不超过 max-samples。
def test_real_download_and_split_e2e(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实下载 UT：默认跳过，按环境变量开启。"""
    monkeypatch.setenv("RUN_REAL_DOWNLOAD_UT", "1")
    if os.environ.get("RUN_REAL_DOWNLOAD_UT") != "1":
        pytest.skip("默认跳过真实下载 UT；设置 RUN_REAL_DOWNLOAD_UT=1 后执行。")

    pytest.importorskip("datasets")

    out_dir = Path("datasets") / "live_ut"
    out_train = out_dir / "train.jsonl"
    out_eval = out_dir / "eval.jsonl"

    # 可通过环境变量调大真实下载 UT 的样本量。
    max_samples = int(os.environ.get("RUN_REAL_DOWNLOAD_MAX_SAMPLES", "30000"))
    if max_samples <= 1:
        raise ValueError("RUN_REAL_DOWNLOAD_MAX_SAMPLES 必须大于 1。")

    out_dir.mkdir(parents=True, exist_ok=True)

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

    assert out_train.exists(), "真实下载后应生成 train.jsonl"
    assert out_eval.exists(), "真实下载后应生成 eval.jsonl"

    train_count = len(out_train.read_text(encoding="utf-8").strip().splitlines())
    eval_count = len(out_eval.read_text(encoding="utf-8").strip().splitlines())

    assert train_count > 0
    assert eval_count > 0
    assert train_count + eval_count <= max_samples

