---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
license: mit
pipeline_tag: text-generation
tags:
- clinical-ai
- medical-qa
- usmle
- qlora
- 4bit
---

# 🩺 Clinical AI 7B QLoRA Adapter (Qwen2.5-7B-Instruct)

This repository contains a 4-bit NormalFloat4 (NF4) Low-Rank Adaptation (QLoRA) adapter fine-tuned on **4,000 USMLE clinical reasoning cases** from the MedQA benchmark.

## Model Details
- **Base Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Adapter Type:** PEFT / LoRA ($r=16, \alpha=32$, dropout=$0.05$)
- **Quantization:** BitsAndBytes 4-bit NormalFloat4 (NF4) + Double Quantization
- **Target Modules:** All linear attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Trainable Parameters:** $20,185,088$ / $7,635,800,064$ ($0.26\%$ of total parameters)
- **Training Strategy:** SFT with Strict Completion-Only Loss Masking (`label=-100` on prompt tokens)
- **Hardware:** NVIDIA Tesla T4 GPU (16GB VRAM, CUDA)

## Evaluation Benchmark (1,000 Held-Out Cases)
- **Diagnostic Option Match Accuracy:** $62.00\%$ ($+33.50\text{ pp}$ improvement over Base 7B at $28.50\%$)
- **Completion Perplexity:** $1.48$ (vs. $4.12$ on Base 7B)
- **Peak Training VRAM:** $7.82\text{ GB}$ (measured via `reset_peak_memory_stats()`)
