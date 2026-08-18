# 🩺 Domain-Specific Clinical Question-Answering LLM Pipeline
### Parameter-Efficient Fine-Tuning (4-Bit NF4 QLoRA & LoRA) on USMLE-Style Clinical Reasoning Cases

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/er3dedrw44i/clinical-llm-finetuning)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/er3dedrw44i/clinical-llm-finetuning/blob/main/qlora_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview & Centerpiece Scope
This repository implements an end-to-end machine learning engineering pipeline for **domain-specific clinical question answering and diagnostic classification**. It fine-tunes open-source foundation models on **5,000 USMLE-style clinical reasoning cases from the MedQA benchmark**.

The project establishes two distinct, reproducible experimental tracks:
1. **Experiment B (Centerpiece — Cloud 4-Bit NF4 QLoRA Track):** Single-GPU 4-bit quantized fine-tuning of `Qwen2.5-7B-Instruct` using `BitsAndBytes` NormalFloat4 (NF4) quantization + PagedAdamW 8-bit optimizer, reducing theoretical base-weight footprint from $\approx 16\text{ GB} \rightarrow 4.5\text{ GB}$ ($\approx 72\%$ compression) and enabling full training under $8.5\text{ GB}$ peak VRAM on an NVIDIA Tesla T4.
2. **Experiment A (Local Apple Silicon Baseline Track):** Prototyping, completion-only loss masking, unit test verification, and first-token latency profiling ($38.8\text{ ms}$) on `Qwen2.5-0.5B-Instruct`.

---

## 🔬 Centerpiece Results & Statistical Evaluation (1,000 Held-Out Cases)

Evaluated strictly on **1,000 unseen test cases (`test.jsonl`)** with **0% exact-string SHA-256 data leakage** and zero base model adapter contamination (`with model.disable_adapter():`):

```text
========================================================================================
                     7B BENCHMARK EVALUATION & STATISTICAL REPORT
========================================================================================
  Metric                           Base 7B             QLoRA 7B            Difference
  --------------------------------------------------------------------------------------
  Diagnostic Accuracy (%)          28.50% [25.7-31.4]  62.00% [58.9-65.0]  +33.50 pp
  Completion Perplexity (PPL)      4.12                1.48                -2.64
  Observed Peak Training VRAM      N/A                 7.82 GB on T4       Fits on free GPU
========================================================================================
```

### 📊 Paired Error Transition Matrix (`results/error_analysis.json`):
* **Wrong $\rightarrow$ Correct ($+35.0\%$):** QLoRA resolved 350 baseline clinical diagnostic errors.
* **Correct $\rightarrow$ Correct ($27.0\%$):** Preserved core medical knowledge on 270 questions.
* **Correct $\rightarrow$ Wrong ($1.5\%$):** Minor regression observed on 15 complex edge cases.
* **Wrong $\rightarrow$ Wrong ($36.5\%$):** 365 challenging multi-specialty diagnostic dilemmas.

---

## 📊 Dataset Diversity & Telemetry Audit
Curated from `medalpaca/medical_meadow_medqa` (5,000 clean records, 5.4 MB) and deterministically partitioned into **persistent `train.jsonl` (4,000 training cases)** and **strictly held-out `test.jsonl` (1,000 evaluation cases)**:

```text
======================================================================
        📊 DEEP DATASET DIVERSITY & TELEMETRY AUDIT
======================================================================
  • Total Records:            5,000 USMLE Cases (train.jsonl / test.jsonl)
  • Training Cases:           4,000 cases (train.jsonl)
  • Held-Out Evaluation:      1,000 cases (test.jsonl - 0% exact-string SHA-256 leakage)
  • Input Word Count:         Mean = 139.3 words | Median = 135 words
  • Unique Vocabulary Tokens: 46,236 tokens (796,331 total words)
  • Heuristic Specialty:      Gastroenterology (69.7%), Cardiology (31.3%),
    (Multi-label keyword      Endocrinology/Renal (27.8%), Pulmonology (27.1%),
     matching; overlapping)   Infectious Disease (23.9%), Pharmacology (22.6%)
  • Option Distribution:      A (20.0%), B (21.3%), C (20.1%), D (20.6%), E (18.1%)
======================================================================
```

---

## 🏗️ The 8-Step Modular Engineering Pipeline

```
  1. [DATASET]           ──► 5,000 USMLE cases split into persistent train.jsonl (4k) & test.jsonl (1k)
  2. [DATA PREPARATION]  ──► SHA-256 deduplication (0% exact leakage), syntax validation, seed (42)
  3. [MODEL LOADER]      ──► Hardware-aware routing (Apple Metal MPS vs. NVIDIA CUDA 4-bit NF4)
  4. [PEFT / LoRA]       ──► Low-Rank Adapter injection (r=16, alpha=32, target: attention & MLP projection layers)
  5. [SFT TRAINING]      ──► Completion-only loss masking (-100) with safe prompt truncation (max_len=512)
  6. [EVALUATION]        ──► Diagnostic Option Match Accuracy (%) & Completion PPL on test.jsonl
  7. [BENCHMARKING]      ──► First-token latency (38.8ms) and throughput (35.4 tok/s) on Apple Silicon
  8. [WEIGHT FUSION]     ──► merge_and_unload() standalone weight consolidation + Streamlit Web App
```

---

## ⚡ 4-Bit NF4 QLoRA Configuration & Memory Math

```python
from transformers import BitsAndBytesConfig

# Hardware-aware 4-bit NormalFloat4 (NF4) Config
compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)
```

### Parameter & Memory Breakdown:
* **Theoretical Base 7B Model in FP16:** $\approx 16.0\text{ GB}$ theoretical unquantized footprint.
* **Quantized 7B Model in 4-bit NF4:** $\approx 4.5\text{ GB}$ VRAM ($\approx 72\%$ weight compression).
* **Trainable Parameters:** $20,185,088$ / $7,635,800,064$ ($0.26\%$ trainable across attention and MLP projection layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Observed Peak Training VRAM:** **`7.82 GB`** on NVIDIA Tesla T4 GPU (measured via `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`).

---

## 🚀 Quickstart & One-Command Execution

```bash
# 1. Clone Repository & Install Pinned Dependencies
git clone https://github.com/er3dedrw44i/clinical-llm-finetuning.git
cd clinical-llm-finetuning
make install

# 2. Run Automated Unit Test Suite (8 tests passing)
make test

# 3. Audit Dataset & Token Length Coverage
make audit

# 4. Run Centerpiece 7B Evaluation
make eval-7b

# 5. Launch Streamlit Clinical Decision Support UI
make app
```

---

## 📄 Defensible Resume Bullet Points

* **Parameter-Efficient Fine-Tuning:** Fine-tuned open-source LLMs on 4,000 USMLE clinical reasoning cases from MedQA using 4-bit NF4 QLoRA and LoRA adapters ($r=16, \alpha=32$, targeting attention and MLP projection layers), reducing active trainable parameters to $0.26\%$.
* **Quantization & Memory Optimization:** Applied BitsAndBytes 4-bit NF4 and double quantization to compress 7B base weights from $\approx 16\text{ GB} \rightarrow 4.5\text{ GB}$ ($\approx 72\%$ reduction), enabling single-GPU training under $7.82\text{ GB}$ peak VRAM on NVIDIA Tesla T4.
* **Completion-Only Loss Masking:** Implemented assistant-only cross-entropy loss masking (`label=-100`) with safe prompt truncation, preventing prompt token memorization and preserving general reasoning capabilities.
* **Evaluation & Statistical Profiling:** Evaluated base vs. fine-tuned models on a strictly held-out 1,000-case test set, achieving a $+33.5\text{ pp}$ gain in diagnostic classification accuracy ($28.5\% \rightarrow 62.0\%$, $95\%\text{ CI: } [58.9, 65.0]$) and reducing completion perplexity from $4.12 \rightarrow 1.48$; benchmarked local Apple Silicon inference at $38.8\text{ ms}$ first-token generation latency.

---

## 📜 License
Released under the [MIT License](LICENSE).
