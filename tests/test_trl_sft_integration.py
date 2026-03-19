from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).parent.parent))

from src.trl_path.train_sft import run_sft


@pytest.fixture
def dummy_data_dir(tmp_path):
    train_file = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    
    # Create sufficient dummy data for a micro batch
    data = []
    for i in range(10):
        data.append(f'{{"instruction": "ins{i}", "input": "", "output": "out{i}"}}')
        
    with open(train_file, "w") as f:
        f.write("\n".join(data))
    
    with open(eval_file, "w") as f:
        f.write("\n".join(data[:2])) # 2 eval samples

    return train_file, eval_file


def test_sft_integration_real_loop(dummy_data_dir, tmp_path):
    """
    Test the full SFT training pipeline with a tiny initialized model.
    No Mocks - Real Transformers/TRL logic.
    """
    train_file, eval_file = dummy_data_dir
    output_dir = tmp_path / "integration_output"

    # 1. Initialize a tiny model (Llama-style config)
    # We use Llama config because it's standard and works well with default target_modules
    config = AutoConfig.from_pretrained("gpt2") 
    config.vocab_size = 50300 # Tiny vocab but enough for gpt2 tokenizer ids
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 32
    
    # Initialize model with random weights
    model = AutoModelForCausalLM.from_config(config)
    
    # 2. Initialize a tokenizer (use gpt2 valid files, but patch vocab size logic if needed)
    # Actually, using gpt2 tokenizer is safest as it doesn't require download if cached, or small download.
    # To be totally offline safe, we'd mock it, but integration test implies real objects.
    # Alternatively, construct a tokenizer from scratch? 
    # Let's try loading gpt2 tokenizer (fast).
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except:
        pytest.skip("Requires internet or cached gpt2 tokenizer")

    # 3. Setup Config
    cfg = {
        "model": {
            "model_name_or_path": "gpt2-tiny-random", # Just a placeholder name
            "trust_remote_code": False,
            "use_fp16": False
        },
        "data": {
            "train_file": str(train_file),
            "eval_file": str(eval_file),
            "max_seq_length": 32 
        },
        "lora": {
            "r": 2,
            "alpha": 4,
            "dropout": 0.05,
            "target_modules": ["c_attn"] # GPT2 attention layer name
        },
        "training": {
            "output_dir": str(output_dir),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-3,
            "logging_steps": 1,
            "save_steps": 10, # Don't save intermediate
            "eval_steps": 10,
            "save_total_limit": 1,
            "lr_scheduler_type": "constant",
            "load_best_model_at_end": False
        },
        "experiment": {
            "run_name": "integration-test"
        }
    }

    args = argparse.Namespace(
        dry_run=False,
        report_to="none",
        resume_from_checkpoint=None,
        run_name="integration-run",
        use_cpu=True # Force CPU
    )

    # 4. Run SFT (injecting model and tokenizer)
    # Note: run_sft will wrap model in PeftModel automatically
    run_sft(args, cfg, model=model, tokenizer=tokenizer)

    # 5. Asset Output
    # Check for Adapter weights
    expected_files = ["adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json"]
    for f in expected_files:
        assert (output_dir / f).exists(), f"Missing output file: {f}"

