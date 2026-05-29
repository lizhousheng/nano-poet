# nano-poet — 8-Week From-Scratch LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://docs.astral.sh/uv/)
[![Course](https://img.shields.io/badge/Course-8%20Weeks-green.svg)](./courses/)

English · [简体中文](./README.md)

> A spiritual descendant of [nanoGPT](https://github.com/karpathy/nanoGPT). Trains on **the complete Tang/Song poetry corpus** instead of Shakespeare. Walks you from a character-level Bigram baseline all the way to QLoRA fine-tuning of Qwen-1.5B over 8 weeks, with a course notebook per week.

## What you build

| Week | Topic | Model | val loss |
|---|---|---|---|
| 1 | Data pipeline + char tokenizer | — | — |
| 2 | Bigram → 6-layer Transformer | 0.79M → 6.37M | 5.43 → 4.14 |
| 3 | LR schedule + AMP + grad clip | 6.37M | ~4.21 |
| 4 | Checkpoint + resume + TensorBoard | 6.37M | ~4.21 |
| 5 | Scale-up: +Song poetry, 25M params, vocab 14k | 25M | 4.03 |
| 6 | SFT instruction tuning (title + style → poem) | 25M | 3.55 |
| 7 | DPO preference alignment (rule-based auto-scoring) | 25M | DPO loss < 0.69 |
| 8 | QLoRA fine-tune Qwen2.5-1.5B (4.3M trainable / 1.5B total) | Qwen + LoRA | — |

## Sample outputs (same kind of prompt across versions)

| Stage | Output snippet |
|---|---|
| Bigram | `月、之之、其...` (char-level random) |
| v0.3 (6-layer, Week 2) | `月光透碧簾，輕雲度遠峰。客舟橫水靜，孤雁過天空。` (rhyme + imagery) |
| v0.9 SFT (Week 6) | `春雨潤晴川，東風拂柳煙。新苗承露重，遠岫接雲連。` |
| Qwen + LoRA (Week 8) | `春雨绵绵润万家，柳丝轻拂燕双斜。一犁新水耕烟绿，几树残红傍水涯。` |

## Repository structure

```
.
├── configs/           Hyperparameters + paths (one file)
├── data/              Data pipeline (download / merge / clean / SFT / DPO prep)
├── tokenizer/         Char-level CharTokenizer (+ v3 with special tokens)
├── model/             Bigram, MiniGPT v01-v03 (shared layers in layers.py)
├── train/             Training loops + utilities (LR schedule, checkpoint, DPO loss)
├── inference/         Generation CLIs (basic / SFT / DPO comparison)
├── evaluation/        pypinyin-based auto-scorer for DPO
├── qlora/             Week 8: load Qwen, prep data, LoRA train, generate
├── courses/           8 .ipynb course notebooks (one per week)
├── checkpoints/       Saved model weights (gitignored content)
└── scripts/           env_check + course builder scripts
```

## Quick start (5 minutes to v0.3)

```bash
# Install uv (Python package manager from Astral, ~10x faster than pip)
# Mac / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Sync deps (creates .venv/, installs torch + tensorboard + pypinyin + tqdm)
uv sync

# Optional Week 8 extras (transformers + peft + trl + bitsandbytes + ...)
uv sync --extra qlora

# Activate venv
source .venv/bin/activate                  # Mac / Linux
.venv\Scripts\Activate.ps1                 # Windows PowerShell

# Run Week 1 data pipeline + Week 2 best model
python data/prepare.py                     # ~1 min
python train/train_v03.py                  # ~5 min on T4 → val ~4.14
python inference/generate.py --model v03 --start 月
```

## Why this project

- **All numbers are real.** Every val-loss target was reached on a Colab T4.
- **Built-in failure cases.** v0.1 (single-head attention with bottleneck) is deliberately *worse* than Bigram — to teach when "adding attention ≠ improvement".
- **No HuggingFace magic** for weeks 1-7. You see every gradient with bare PyTorch. Week 8 finally introduces the HF stack so you understand what it abstracts away.
- **8 weeks of pedagogical notebooks**, each with theory + code map + hands-on commands + self-tests + common pitfalls.

## Run in cloud

- **Google Colab** (free T4, recommended outside China) — see Chinese README's *在 Colab 上跑* section
- **ModelScope** (free A10, recommended in China) — see Chinese README's *在 ModelScope 上跑* section

## Course curriculum

See [`courses/`](./courses/) for the 8 standalone notebooks. Each follows the same 7-part structure:
0. This week's goals
1. Prerequisites
2. Core concepts (theory + code snippets)
3. Code map (which files, which functions)
4. Hands-on commands
5. Self-test questions (no answers — answer them in your head)
6. Common pitfalls

Week 2 ships an extra `week2_self_test.md` with 30 Q&A on Transformer fundamentals.

## Related work

- [nanoGPT](https://github.com/karpathy/nanoGPT) — the spiritual ancestor (Shakespeare, single notebook)
- [llm.c](https://github.com/karpathy/llm.c) — Karpathy's C/CUDA port of GPT-2 training
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) — Chinese-finetuned LLaMA

## License

[MIT](./LICENSE)
