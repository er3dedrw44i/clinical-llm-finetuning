---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: mit
pipeline_tag: text-generation
tags:
- clinical-ai
- medical-qa
- usmle
- lora
- peft
---

# 🩺 Clinical AI LoRA Adapter (Qwen2.5-0.5B)

This repository contains a Low-Rank Adaptation (LoRA) adapter fine-tuned on **4,000 USMLE clinical reasoning cases** from the MedQA benchmark.

## Model Details
- **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter Type:** PEFT / LoRA ($r=16, \alpha=32$, dropout=$0.05$)
- **Target Modules:** All linear attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Trainable Parameters:** $8,798,208$ ($1.75\%$ of total parameters)
- **Training Strategy:** SFT with Strict Completion-Only Loss Masking (`label=-100` on prompt tokens)
- **Hardware:** Apple Silicon M5 GPU (`mps`)

## Intended Use & Scope
- **Domain:** Diagnostic option selection and clinical pharmacology triage.
- **Out of Scope:** This model is an experimental research prototype and is **not certified for autonomous clinical decision-making or real-world patient treatment**.

## Evaluation Summary
- **Held-Out Test Set:** 1,000 unseen MedQA clinical cases (`test.jsonl`).
- **Primary Metric:** Diagnostic Option Match Accuracy.
- **Latency:** $38.87\text{ ms}$ first-token generation latency, $35.41\text{ tok/s}$ throughput on Apple Silicon.