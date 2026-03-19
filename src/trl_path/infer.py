from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """解析推理参数，支持可选 LoRA adapter。"""
    parser = argparse.ArgumentParser(description="Inference with Qwen base model and optional LoRA adapter.")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter-path", default="", help="Optional PEFT adapter path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    """加载基座模型（可叠加 LoRA）并执行单轮文本生成。"""
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        # 保证批处理和生成时 pad 行为稳定。
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    if args.adapter_path:
        from peft import PeftModel

        # 推理阶段按需加载 adapter，不改动基座权重。
        model = PeftModel.from_pretrained(model, args.adapter_path)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()

