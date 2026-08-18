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

This directory contains the PEFT configuration for the 4-bit NormalFloat4 (NF4) Low-Rank Adaptation (QLoRA) adapter fine-tuned on **4,000 USMLE clinical reasoning cases** from the MedQA benchmark.

## Model Details
- **Base Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Adapter Type:** PEFT / LoRA ($r=16, \alpha=32$, dropout=$0.05$)
- **Quantization:** BitsAndBytes 4-bit NormalFloat4 (NF4) + Double Quantization
- **Target Modules:** All linear attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Trainable Parameters:** $20,185,088$ / $7,635,800,064$ ($0.26\%$ of total parameters)
- **Training Strategy:** SFT with Strict Completion-Only Loss Masking (`label=-100` on prompt tokens)
- **Hardware:** NVIDIA Tesla T4 GPU (16GB VRAM, CUDA)

## Reproducing Adapter Weights
To reproduce the trained adapter weights (`adapter_model.safetensors`), execute the canonical training pipeline:
1. **Google Colab (Recommended):** Run [`qlora_colab.ipynb`](../qlora_colab.ipynb) on a free Tesla T4 GPU.
2. **Local CUDA GPU:** Run `python3 qlora_train.py` (or `make train`).

The training loop automatically saves the fine-tuned `adapter_model.safetensors` into this directory upon completion.
