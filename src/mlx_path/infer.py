from __future__ import annotations

import argparse
import shlex
import subprocess


def parse_args() -> argparse.Namespace:
    """解析 MLX 推理参数。"""
    parser = argparse.ArgumentParser(description="Run inference via mlx_lm.generate CLI wrapper.")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    """组装 mlx_lm.generate 命令。"""
    cmd = [
        "python",
        "-m",
        "mlx_lm.generate",
        "--model",
        args.model_path,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temp",
        str(args.temperature),
    ]

    if args.adapter_path:
        cmd.extend(["--adapter-path", args.adapter_path])

    return cmd


def main() -> None:
    """打印并执行 MLX 推理命令。"""
    args = parse_args()
    command = build_command(args)

    print("Generated MLX inference command:")
    print(" ".join(shlex.quote(c) for c in command))

    if args.dry_run:
        # dry-run 用于确认命令而不真正执行生成。
        print("Dry run complete.")
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

