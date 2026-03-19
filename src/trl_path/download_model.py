from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析模型下载参数。"""
    parser = argparse.ArgumentParser(description="Download model snapshot from Hugging Face Hub.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", default="models/hf_qwen2_5_0_5b")
    return parser.parse_args()


def main() -> None:
    """下载并缓存模型权重，供 TRL 路径训练和推理使用。"""
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with `pip install huggingface_hub`."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用快照下载可保证 tokenizer 和权重版本一致。
    local_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to: {local_path}")


if __name__ == "__main__":
    main()

