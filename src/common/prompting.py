from __future__ import annotations


# ============================================================
# 【原理说明】指令微调数据格式 (Instruction Tuning Format)
# ============================================================
#
# 指令微调（Instruction Tuning）通过让模型学习"指令 -> 回答"映射，
# 将预训练模型变成能听懂人类意图的对话/任务助手。
#
# 本模块使用的三段式模板：
#
#   ### Instruction:
#   <任务描述，告诉模型要做什么>
#
#   ### Input:        ← 可选，有上下文/素材时才写
#   <补充输入内容>
#
#   ### Response:
#   <期望模型生成的标准答案>
#
# 为什么需要固定模板？
#   1. 训练时：模型通过大量"指令+回答"样本学会格式规律。
#   2. 推理时：用相同模板填入新指令，模型按已学规律生成回答。
#   3. 格式一致性决定了 LoRA adapter 的泛化效果；训练和推理
#      使用不同格式会导致回答质量大幅下降。
#
# 常见格式变体参考：
#   - Alpaca 格式（本项目）：### Instruction / Input / Response
#   - ChatML 格式：<|im_start|>user ... <|im_end|>
#   - Llama-2 格式：[INST] ... [/INST]
# ============================================================


def format_instruction_sample(instruction: str, input_text: str, output_text: str) -> str:
    """将一条指令样本格式化为统一的训练文本。"""
    # 训练前统一清理空白，避免数据里多余换行影响学习目标。
    input_text = input_text.strip()
    parts = [f"### Instruction:\n{instruction.strip()}"]

    # 只有在 input 非空时才写入该段，兼容纯指令任务。
    if input_text:
        parts.append(f"### Input:\n{input_text}")

    parts.append(f"### Response:\n{output_text.strip()}")
    return "\n\n".join(parts)

