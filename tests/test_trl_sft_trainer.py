from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from src.trl_path.train_sft import run_sft


class MockAutoTokenizer:
    pad_token = None
    eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, *args, **kwargs):
        # Return dummy encoded input
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1]
        }
    
    def save_pretrained(self, save_directory):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "tokenizer.json").touch()

    def decode(self, *args, **kwargs):
        return "dummy text"


class MockAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = MagicMock()
        model.config.vocab_size = 100
        # Determine device based on kwargs or default to cpu
        # Thismock needs to simulate model.to()
        model.to.return_value = model
        return model

class MockSFTTrainer:
    def __init__(self, *args, **kwargs):
        pass

    def train(self, resume_from_checkpoint=None):
        pass

    def save_model(self, output_dir):
        # Simulate saving adapter
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "adapter_model.safetensors").touch()


@pytest.fixture
def dummy_data_files(tmp_path):
    train_file = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    
    with open(train_file, "w") as f:
        f.write('{"instruction": "i1", "input": "", "output": "o1"}\n')
        f.write('{"instruction": "i2", "input": "", "output": "o2"}\n')
        
    with open(eval_file, "w") as f:
        f.write('{"instruction": "e1", "input": "", "output": "eo1"}\n')

    return str(train_file), str(eval_file)


@patch("transformers.AutoTokenizer", MockAutoTokenizer)
@patch("transformers.AutoModelForCausalLM", MockAutoModelForCausalLM)
@patch("trl.SFTTrainer", MockSFTTrainer)
def test_sft_training_loop(dummy_data_files, tmp_path):
    """
    Integration test for run_sft using mocks for heavy ML components.
    Verifies that the configuration is parsed correctly and the training loop
    (via mocked Trainer) is executed and artifacts are 'saved'.
    """
    # Import SFTConfig to check if it's used if we spy on args
    from trl import SFTConfig

    train_file, eval_file = dummy_data_files
    output_dir = tmp_path / "output"

    cfg = {
        "model": {
            "model_name_or_path": "dummy-model",
            "trust_remote_code": True,
            "use_fp16": False
        },
        "data": {
            "train_file": train_file,
            "eval_file": eval_file,
            "max_seq_length": 64
        },
        "lora": {
            "r": 4,
            "alpha": 8,
            "dropout": 0.1,
            "target_modules": ["q_proj"]
        },
        "training": {
            "output_dir": str(output_dir),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "logging_steps": 1,
            "save_steps": 1,
            "eval_steps": 1,
            "save_total_limit": 1,
        },
        "experiment": {
            "run_name": "test-run"
        }
    }

    args = argparse.Namespace(
        dry_run=False,
        report_to="none",
        resume_from_checkpoint=None,
        run_name="test-cli-run",
        use_cpu=True  # Force CPU to avoid CUDA checks in tests
    )

    # Run the function
    run_sft(args, cfg)

    # Verify outputs
    assert (output_dir / "adapter_model.safetensors").exists()
    assert (output_dir / "tokenizer.json").exists()

