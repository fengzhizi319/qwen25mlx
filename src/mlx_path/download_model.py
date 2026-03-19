from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析 MLX 路径模型下载参数。"""
    parser = argparse.ArgumentParser(description="Download model snapshot for MLX workflow.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="models/hf_qwen2_5_0_5b")
    return parser.parse_args()


def main() -> None:
    """下载模型快照，供 MLX 转换/训练/推理复用。"""
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with `pip install huggingface_hub`."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 与 TRL 路径共用同一模型缓存，便于对比实验。
    local_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to: {local_path}")


if __name__ == "__main__":
    main()

