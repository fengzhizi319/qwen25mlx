from __future__ import annotations


# ============================================================
# 【原理说明】内存分档与训练参数建议
# ============================================================
#
# 为什么内存是关键限制因素？
#   训练期间显存/内存主要被以下部分占用：
#     1. 模型权重（FP32/FP16）：Qwen2.5-0.5B ≈ 1GB
#     2. 梯度：与权重同等大小
#     3. 优化器状态（AdamW）：权重的 2 倍
#     4. 激活值：与 batch_size × seq_len 正相关
#
#   粗估：总占用 ≈ 模型权重 × (1 + 1 + 2) + 激活值
#   LoRA 大幅减少了 1/2/3 项，但激活值仍随 batch 线性增长。
#
# 分档逻辑：
#   xs (≤8GB)   → 最小 batch，保守样本量，避免 OOM
#   s  (≤16GB)  → 单卡学习型实验
#   m  (≤32GB)  → 标准微调，可跑完整 Alpaca 数据集
#   l  (≤64GB)  → 多轮迭代实验
#   xl (>64GB)  → 大规模数据训练
#
# profile 倍率：
#   safe(0.7) < balanced(1.0) < aggressive(1.35)
#   建议初学者从 balanced 开始，遇到 OOM 切换到 safe。
#
# 梯度累积建议：
#   目标有效 batch = 8
#   accumulation_steps = 8 // batch_size
#   这样无论 batch_size 大小，实际更新方差近似一致。
# ============================================================

import argparse
import json
import platform
import subprocess
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析参数：支持自动探测内存，也支持手工覆盖内存值。"""
    parser = argparse.ArgumentParser(description="按机器内存建议 max-samples 与 batch size。")
    parser.add_argument("--framework", choices=["trl", "mlx"], default="trl")
    parser.add_argument(
        "--memory-gb",
        type=float,
        default=None,
        help="手动指定总内存（GB），用于覆盖自动探测结果。",
    )
    parser.add_argument(
        "--profile",
        choices=["safe", "balanced", "aggressive"],
        default="balanced",
        help="建议策略：safe 更稳，aggressive 更激进。",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出，便于脚本消费。")
    return parser.parse_args()


def _detect_memory_gb_macos() -> float | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        total_bytes = int(out)
        return total_bytes / (1024**3)
    except Exception:
        return None


def _detect_memory_gb_linux() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    mem_kb = int(parts[1])
                    return mem_kb / (1024**2)
    except Exception:
        return None
    return None


def _detect_memory_gb_windows() -> float | None:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullTotalPhys / (1024**3)
    except Exception:
        return None


def detect_total_memory_gb() -> float:
    """自动探测机器总内存，失败时抛出错误提示手工传参。"""
    system_name = platform.system().lower()

    if system_name == "darwin":
        mem = _detect_memory_gb_macos()
    elif system_name == "linux":
        mem = _detect_memory_gb_linux()
    elif system_name == "windows":
        mem = _detect_memory_gb_windows()
    else:
        mem = None

    if mem is None:
        raise RuntimeError("无法自动探测内存，请通过 --memory-gb 手工指定。")
    return mem


def _pick_tier(memory_gb: float) -> str:
    if memory_gb <= 8:
        return "xs"
    if memory_gb <= 16:
        return "s"
    if memory_gb <= 32:
        return "m"
    if memory_gb <= 64:
        return "l"
    return "xl"


def recommend_from_memory_gb(memory_gb: float, framework: str, profile: str) -> dict[str, Any]:
    """根据总内存、框架和策略给出建议参数。"""
    tier = _pick_tier(memory_gb)

    # 不同策略通过倍率放大/缩小样本规模与 batch 大小。
    profile_scale = {"safe": 0.7, "balanced": 1.0, "aggressive": 1.35}[profile]

    if framework == "trl":
        base = {
            "xs": {"max_samples": 2000, "batch_size": 1},
            "s": {"max_samples": 5000, "batch_size": 1},
            "m": {"max_samples": 12000, "batch_size": 2},
            "l": {"max_samples": 25000, "batch_size": 4},
            "xl": {"max_samples": 50000, "batch_size": 8},
        }
    elif framework == "mlx":
        base = {
            "xs": {"max_samples": 4000, "batch_size": 1},
            "s": {"max_samples": 10000, "batch_size": 2},
            "m": {"max_samples": 25000, "batch_size": 4},
            "l": {"max_samples": 50000, "batch_size": 8},
            "xl": {"max_samples": 100000, "batch_size": 12},
        }
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    raw = base[tier]
    max_samples = max(1000, int(raw["max_samples"] * profile_scale))
    batch_size = max(1, int(round(raw["batch_size"] * profile_scale)))

    # TRL 增加梯度累积建议，帮助在小显存下获得更稳定的全局 batch。
    gradient_accumulation_steps = max(1, 8 // batch_size)

    return {
        "framework": framework,
        "profile": profile,
        "memory_gb": round(memory_gb, 2),
        "tier": tier,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "note": "建议值用于起步，训练中请根据实际 OOM/速度继续微调。",
    }


def build_config_snippet(rec: dict[str, Any]) -> dict[str, Any]:
    """生成可直接抄到配置里的片段。"""
    if rec["framework"] == "trl":
        return {
            "dataset_download": {"max_samples": rec["max_samples"]},
            "training": {
                "per_device_train_batch_size": rec["batch_size"],
                "gradient_accumulation_steps": rec["gradient_accumulation_steps"],
            },
        }
    return {
        "dataset_download": {"max_samples": rec["max_samples"]},
        "training": {"batch_size": rec["batch_size"]},
    }


def main() -> None:
    args = parse_args()

    memory_gb = args.memory_gb
    if memory_gb is None:
        memory_gb = detect_total_memory_gb()

    if memory_gb <= 0:
        raise ValueError("内存值必须大于 0。")

    rec = recommend_from_memory_gb(memory_gb, args.framework, args.profile)
    snippet = build_config_snippet(rec)

    if args.json:
        print(json.dumps({"recommendation": rec, "config_snippet": snippet}, ensure_ascii=False, indent=2))
        return

    print("=== 自动建议结果 ===")
    print(f"framework: {rec['framework']}")
    print(f"profile:   {rec['profile']}")
    print(f"memory_gb: {rec['memory_gb']}")
    print(f"tier:      {rec['tier']}")
    print(f"max_samples: {rec['max_samples']}")
    print(f"batch_size:  {rec['batch_size']}")
    if rec["framework"] == "trl":
        print(f"gradient_accumulation_steps: {rec['gradient_accumulation_steps']}")

    print("\n=== 可粘贴配置片段(JSON) ===")
    print(json.dumps(snippet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

