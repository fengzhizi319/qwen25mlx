# Qwen2.5-0.5B Learning Lab (TRL + MLX)

> 🇨🇳 [中文版文档 → README_zh.md](README_zh.md)

A complete learning project for:
- downloading `Qwen/Qwen2.5-0.5B`
- understanding data formatting and pipelines
- fine-tuning/training with Hugging Face TRL (LoRA)
- fine-tuning/training with Apple MLX (LoRA)
- running inference on both paths

This project is organized for study first: small scripts, clear config files, and smoke tests.

## 1) Project layout

```text
qwen25/
  configs/
    trl_sft.yaml
    trl_sft_downloaded.yaml
    mlx_lora.yaml
  data/
    minimal/
      train.jsonl
      eval.jsonl
  scripts/
    demo_runner.py
  src/
    common/
      data.py
      download_dataset.py
      recommend_training.py
      prompting.py
    trl_path/
      download_model.py
      train_sft.py
      infer.py
    mlx_path/
      download_model.py
      prepare_data.py
      train_lora.py
      infer.py
  tests/
    test_data_pipeline.py
    test_dataset_download.py
    test_dataset_download_live.py
    test_recommend_training.py
    test_prompting.py
    test_mlx_commands.py
    test_mlx_prepare.py
    test_trl_train_utils.py
  pyproject.toml
  requirements.txt
  requirements-trl.txt
  requirements-mlx.txt
  README_beginner.md
  README_advanced.md
  README_zh.md
```

## 2) Environment setup

Python 3.10+ is recommended.

```bash
cd /Users/charles/Documents/AI/qwen25
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

For TRL path:

```bash
python -m pip install -r requirements-trl.txt
```

For MLX path (Apple Silicon):

```bash
python -m pip install -r requirements-mlx.txt
```

## 3) Run tests and smoke checks

```bash
pytest -q
python scripts/demo_runner.py
```

## 4) Download Qwen2.5-0.5B

The default Hugging Face model id is `Qwen/Qwen2.5-0.5B`.

```bash
python -m src.trl_path.download_model --model-id Qwen/Qwen2.5-0.5B --output-dir models/hf_qwen2_5_0_5b
```

For MLX flow, you can also cache the same model first:

```bash
python -m src.mlx_path.download_model --model-id Qwen/Qwen2.5-0.5B --output-dir models/hf_qwen2_5_0_5b
```

## 5) Data preparation notes

Input data uses JSONL records with `instruction`, `input`, and `output`.
The training script formats each sample into an instruction-style prompt.

Example record:

```json
{"instruction": "Translate to Chinese", "input": "hello", "output": "你好"}
```

Download a larger training dataset (recommended for real fine-tuning practice):

```bash
python -m src.common.download_dataset \
  --dataset yahma/alpaca-cleaned \
  --split train \
  --max-samples 5000 \
  --eval-ratio 0.02 \
  --out-train data/downloaded/train.jsonl \
  --out-eval data/downloaded/eval.jsonl
```

If your dataset uses different column names, map them with:

```bash
python -m src.common.download_dataset \
  --dataset some_org/some_dataset \
  --instruction-col prompt \
  --input-col context \
  --output-col response
```

Auto-suggest `max-samples` and batch size by machine memory:

```bash
python -m src.common.recommend_training --framework trl --profile balanced
python -m src.common.recommend_training --framework mlx --profile balanced
python -m src.common.recommend_training --framework trl --memory-gb 32 --json
```

## 6) TRL LoRA training

Update `configs/trl_sft.yaml` as needed, then run:

```bash
python -m src.trl_path.train_sft --config configs/trl_sft.yaml
```

Train with downloaded dataset config:

```bash
python -m src.trl_path.train_sft --config configs/trl_sft_downloaded.yaml
```

Dry run without loading model/training:

```bash
python -m src.trl_path.train_sft --config configs/trl_sft.yaml --dry-run
```

Resume training from latest checkpoint automatically:

```bash
python -m src.trl_path.train_sft --config configs/trl_sft.yaml --resume-from-checkpoint auto
```

Enable W&B logging (optional):

```bash
python -m pip install wandb
python -m src.trl_path.train_sft \
  --config configs/trl_sft.yaml \
  --report-to wandb \
  --wandb-project qwen25-lab \
  --run-name qwen25-exp-01
```

Run inference from base model + LoRA adapter:

```bash
python -m src.trl_path.infer \
  --model-path Qwen/Qwen2.5-0.5B \
  --adapter-path outputs/trl_lora_adapter \
  --prompt "Explain LoRA in one paragraph." \
  --max-new-tokens 120
```

## 7) MLX LoRA training

`src/mlx_path/train_lora.py` and `src/mlx_path/infer.py` are wrappers around `mlx_lm` commands.
This keeps learning logic explicit and easy to modify.

Dry run (print generated command only):

```bash
python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml --dry-run
```

The wrapper auto-converts `data/minimal/*.jsonl` into `data/mlx/train.jsonl` and `data/mlx/valid.jsonl`.
If you already prepared MLX data, skip that step:

```bash
python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml --skip-prepare-data
```

Actual run:

```bash
python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml
```

Generate with MLX:

```bash
python -m src.mlx_path.infer \
  --model-path Qwen/Qwen2.5-0.5B \
  --adapter-path outputs/mlx_adapters \
  --prompt "Write a short note about transfer learning." \
  --max-tokens 120
```

## 8) Two learning tracks in README

You can also follow dedicated track docs:

- Beginner: `README_beginner.md`
- Advanced: `README_advanced.md`

### Track A: Beginner route (minimum runnable loop)

1. **Understand data format first**
   - Run `python scripts/demo_runner.py`
   - Read `src/common/prompting.py` and `src/common/data.py`
2. **Run a safe dry-run**
   - `python -m src.trl_path.train_sft --config configs/trl_sft.yaml --dry-run`
   - Confirm train/eval sample counts and config loading are normal
3. **Do a tiny TRL LoRA training**
   - `python -m src.trl_path.train_sft --config configs/trl_sft.yaml`
   - Observe outputs under `outputs/trl_lora_adapter`
4. **Run inference and compare**
   - Use `src/trl_path/infer.py` with and without adapter
   - Note how answer style changes after LoRA
5. **Try MLX command flow**
   - `python -m src.mlx_path.train_lora --config configs/mlx_lora.yaml --dry-run`
   - Understand how wrapper scripts translate config to CLI

### Track B: Advanced route (full experiment workflow)

1. **Download a larger training dataset**
   - Run `python -m src.common.download_dataset ...`
   - Use `data/downloaded/train.jsonl` and `data/downloaded/eval.jsonl`
2. **Switch to downloaded-data config**
   - Run `python -m src.trl_path.train_sft --config configs/trl_sft_downloaded.yaml`
   - Tune `max_seq_length`, `learning_rate`, `eval_steps`, `save_steps`
3. **Enable experiment tracking and resume**
   - Use `--report-to wandb` and set `--wandb-project`
   - Resume via `--resume-from-checkpoint auto`
4. **Cross-framework comparison (TRL vs MLX)**
   - Run both dry-run and small training loops
   - Compare throughput, memory usage, and output quality
5. **Drive changes with tests**
   - Extend tests under `tests/` before changing templates or configs
   - Keep `pytest -q` green while iterating

## 9) Learning-oriented tests

Use these tests to understand each module:

- `tests/test_data_pipeline.py`: JSONL parsing, empty line skip, missing key validation.
- `tests/test_prompting.py`: prompt template formatting and whitespace normalization.
- `tests/test_dataset_download.py`: downloaded dataset mapping and parameter validation.
- `tests/test_dataset_download_live.py`: real network download E2E test (env-gated).
- `tests/test_recommend_training.py`: memory-based parameter suggestion tests.
- `tests/test_mlx_commands.py`: MLX train/infer command construction checks.
- `tests/test_trl_train_utils.py`: checkpoint auto-resume and report backend resolution.
- `tests/test_mlx_prepare.py`: data conversion into MLX train/valid files.

Run all tests:

```bash
pytest -q
```

Run real download UT only (optional):

```bash
cd /Users/charles/Documents/AI/qwen25
source .venv/bin/activate
RUN_REAL_DOWNLOAD_UT=1 pytest -q tests/test_dataset_download_live.py
```

## 10) Troubleshooting

- If model download is slow, set a mirror/proxy for Hugging Face in your shell.
- On macOS without CUDA, TRL runs on CPU/MPS and can be slow; keep batch/steps small.
- If `mlx_lm` command fails, verify `requirements-mlx.txt` is installed and Python points to the active venv.

## 11) License and model usage

Check the original model card and license for `Qwen/Qwen2.5-0.5B` on Hugging Face before production or redistribution.


