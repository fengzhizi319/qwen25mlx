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

    '''
    datasets是HuggingFace提供的一个Python数据集库，常用于：
    。下载公开数据集
    。读取本地 / 远程数据
    。对数据做切分、过滤、映射、打乱
    。方便训练、评估、测试机器学习模型
    它的核心特点是：
    。统一接口
    。支持流式 / 懒加载    
    。和Hugging Face生态集成很好很适合
    。NLP / LLM训练前的数据处理
    '''
    # 依赖存在 → 继续测试
    # 依赖不存在 → 自动跳过测试
    pytest.importorskip("datasets")
    out_dir = Path("datasets") / "live_ut"
    out_train = out_dir / "train.jsonl"
    out_eval = out_dir / "eval.jsonl"

    # 可通过环境变量调大真实下载 UT 的样本量。
    max_samples = int(os.environ.get("RUN_REAL_DOWNLOAD_MAX_SAMPLES", "30000"))
    if max_samples <= 1:
        raise ValueError("RUN_REAL_DOWNLOAD_MAX_SAMPLES 必须大于 1。")

    out_dir.mkdir(parents=True, exist_ok=True)
    '''
    1) monkeypatch.setattr 是什么
       monkeypatch.setattr 是 pytest 提供的测试工具，用来：
       在测试期间临时替换某个对象的属性/方法，测试结束后自动恢复。
       它常用于：
       替换全局变量
       mock 掉函数
       修改 sys.argv
       替换环境里某个模块的行为
    2) 把当前测试进程里的 sys.argv 临时替换成你给的命令行参数列表。
    '''
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

