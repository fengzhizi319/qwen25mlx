from __future__ import annotations

from src.common.prompting import format_instruction_sample


# 教学注释块
# 测试目的: 当 input 存在时，模板必须包含 Input 段。
# 关键断言: Instruction/Input/Response 三段齐全。
def test_format_instruction_sample_with_input() -> None:
    """有 input 时应包含 Input 段落。"""
    text = format_instruction_sample(
        instruction="Translate to Chinese",
        input_text="hello",
        output_text="你好",
    )
    assert "### Instruction:" in text
    assert "### Input:" in text
    assert "### Response:" in text


# 教学注释块
# 测试目的: 当 input 为空时，模板不应出现 Input 段。
# 关键断言: 仅保留 Instruction 与 Response。
def test_format_instruction_sample_without_input() -> None:
    """无 input 时不应生成 Input 段落。"""
    text = format_instruction_sample(
        instruction="Define tokenizer",
        input_text="",
        output_text="Tokenizer maps text to tokens.",
    )
    assert "### Input:" not in text
    assert "### Response:" in text


# 教学注释块
# 测试目的: 验证前后空白会被清理，避免脏文本进入训练。
# 关键断言: 指令、输入、输出中的空白都被规范化。
def test_format_instruction_sample_strips_whitespace() -> None:
    """学习测试：前后空白会被清理，避免训练文本脏数据。"""
    text = format_instruction_sample(
        instruction="  Explain tokenizer  ",
        input_text="  text  ",
        output_text="  A tokenizer maps text to ids.  ",
    )
    assert "Explain tokenizer" in text
    assert "\ntext\n" in f"\n{text}\n"
    assert "A tokenizer maps text to ids." in text


