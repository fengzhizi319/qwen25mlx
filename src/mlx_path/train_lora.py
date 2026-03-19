from __future__ import annotations


# ============================================================
# 【原理说明】Apple MLX 框架与 LoRA 在 Apple Silicon 上的优势
# ============================================================
#
# MLX 是 Apple 专为 Apple Silicon（M 系列芯片）设计的机器学习框架。
# 与 PyTorch/TRL 的核心区别：
#
#   架构层面：
#     - 统一内存（Unified Memory）：CPU 与 GPU 共享同一物理内存，
#       无需数据在设备间拷贝，省去 torch.to(device) 的开销。
#     - 延迟求值（Lazy Evaluation）：计算图先构建再执行，
#       更容易做 kernel 融合优化。
#
#   LoRA 在 MLX 上的流程（mlx_lm.lora）：
#     1. 加载 HuggingFace 格式的权重，原地转换为 MLX 张量。
#     2. 在注意力层注入 LoRA A/B 矩阵（与 PEFT 原理相同）。
#     3. 使用 SGD/AdamW 在统一内存上训练 adapter。
#     4. 保存 adapter 到 --adapter-path，格式与 PEFT 兼容。
#
#   Apple Silicon 上 MLX vs TRL(MPS) 对比：
#     ┌──────────────┬──────────────┬──────────────┐
#     │              │ MLX          │ TRL (MPS)    │
#     ├──────────────┼──────────────┼──────────────┤
#     │ 内存利用率   │ 高（统一内存）│ 较低（显存限制）│
#     │ 小模型速度   │ 快           │ 中等          │
#     │ 生态成熟度   │ 成长中       │ 成熟          │
#     │ 调试便利性   │ 较难         │ 较方便        │
#     └──────────────┴──────────────┴──────────────┘
#
# 本脚本是 mlx_lm.lora 命令的配置化封装，
# 将 YAML 配置翻译成等价的 CLI 参数并执行。
# ============================================================

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """解析 MLX LoRA 训练参数。"""
    parser = argparse.ArgumentParser(description="Run MLX LoRA training via mlx_lm CLI wrapper.")
    parser.add_argument("--config", default="configs/mlx_lora.yaml")
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_command(cfg: dict) -> list[str]:
    """组装 mlx_lm.lora 命令行参数。"""
    model_path = str(cfg["model"]["model_path"])
    train_cfg = cfg["training"]

    cmd = [
        "python",
        "-m",
        "mlx_lm.lora",
        "--model",
        model_path,
        "--train",
        "--data",
        str(train_cfg["data_dir"]),
        "--adapter-path",
        str(train_cfg["adapter_path"]),
        "--iters",
        str(train_cfg["iters"]),
        "--batch-size",
        str(train_cfg["batch_size"]),
        "--learning-rate",
        str(train_cfg["learning_rate"]),
        "--steps-per-eval",
        str(train_cfg["steps_per_eval"]),
        "--steps-per-report",
        str(train_cfg["steps_per_report"]),
        "--save-every",
        str(train_cfg["save_every"]),
    ]
    return cmd


def main() -> None:
    """按配置执行数据准备与 MLX LoRA 训练命令。"""
    args = parse_args()
    cfg = load_yaml(args.config)

    if not args.skip_prepare_data:
        # 训练前先将通用 JSONL 转成 MLX 约定的 train/valid 文件。
        prep_cmd = [
            "python",
            "-m",
            "src.mlx_path.prepare_data",
            "--train-file",
            str(cfg["training"]["source_train_file"]),
            "--eval-file",
            str(cfg["training"]["source_eval_file"]),
            "--output-dir",
            str(cfg["training"]["data_dir"]),
        ]
        print("Generated MLX data prep command:")
        print(" ".join(shlex.quote(c) for c in prep_cmd))
        if not args.dry_run:
            subprocess.run(prep_cmd, check=True)

    command = build_command(cfg)

    print("Generated MLX training command:")
    print(" ".join(shlex.quote(c) for c in command))

    if args.dry_run:
        # dry-run 只打印命令，便于学习和检查参数。
        print("Dry run complete.")
        return

    Path(cfg["training"]["adapter_path"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()


