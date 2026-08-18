# 🩺 Domain-Specific Clinical Question-Answering LLM Pipeline
### Parameter-Efficient Fine-Tuning (LoRA & 4-Bit NF4 QLoRA) on USMLE-Style Clinical Cases

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/er3dedrw44i/clinical-llm-finetuning)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/er3dedrw44i/clinical-llm-finetuning/blob/main/qlora_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview & Scope
This repository implements an end-to-end machine learning pipeline for **domain-specific clinical question answering and diagnostic classification**. It fine-tunes open-source autoregressive foundation models (`Qwen2.5`) on **USMLE-style clinical reasoning cases from the MedQA benchmark**.

The project maintains two distinct, reproducible experimental tracks:
1. **Experiment A (Local Apple Silicon MPS Baseline):** Prototyping, completion-only SFT loss masking (`-100`), evaluation harness, and latency profiling on `Qwen2.5-0.5B-Instruct`.
2. **Experiment B (Cloud 4-Bit NF4 QLoRA Track):** Single-GPU 4-bit quantized fine-tuning of `Qwen2.5-7B-Instruct` using `BitsAndBytes` NormalFloat4 (NF4) quantization, reducing theoretical base-weight footprint from $\approx 16\text{ GB} \rightarrow 4.5\text{ GB}$ ($\approx 72\%$ compression).

---

## 📊 Dataset Telemetry & Diversity Audit
Curated from `medalpaca/medical_meadow_medqa` (5,000 clean records, 5.4 MB) and deterministically partitioned into **persistent `train.jsonl` (4,000 training cases)** and **strictly held-out `test.jsonl` (1,000 evaluation cases)**:

```text
======================================================================
        📊 DEEP DATASET DIVERSITY & TELEMETRY AUDIT
======================================================================
  • Total Records:            5,000 USMLE Cases (train.jsonl / test.jsonl)
  • Training Cases:           4,000 cases (train.jsonl)
  • Held-Out Evaluation:      1,000 cases (test.jsonl - 0% hash leakage)
  • Input Word Count:         Mean = 139.3 words | Median = 135 words
  • Unique Vocabulary Tokens: 46,236 tokens (796,331 total words)
  • Specialty Coverage:       Gastroenterology (69.7%), Cardiology (31.3%),
                              Endocrinology/Renal (27.8%), Pulmonology (27.1%),
                              Infectious Disease (23.9%), Pharmacology (22.6%)
  • Option Distribution:      A (20.0%), B (21.3%), C (20.1%), D (20.6%), E (18.1%)
======================================================================
```

---

## 🏗️ The 8-Step Modular Engineering Pipeline

```
  1. [DATASET]           ──► 5,000 USMLE cases split into persistent train.jsonl (4k) & test.jsonl (1k)
  2. [DATA PREPARATION]  ──► SHA-256 deduplication, syntax validation, deterministic seed (42)
  3. [MODEL LOADER]      ──► Hardware-aware routing (Apple Metal MPS vs. NVIDIA CUDA 4-bit NF4)
  4. [PEFT / LoRA]       ──► Low-Rank Adapter injection (r=16, alpha=32, target: q, k, v, o, gate, up, down)
  5. [SFT TRAINING]      ──► Completion-only loss masking (-100 on prompt tokens)
  6. [EVALUATION]        ──► Diagnostic Option Match Accuracy (%) & Completion PPL on test.jsonl
  7. [BENCHMARKING]      ──► First-token latency (38.8ms) and throughput (35.4 tok/s) on Apple Silicon
  8. [WEIGHT FUSION]     ──► merge_and_unload() standalone weight consolidation + Streamlit Web App
```

---

## ⚡ 4-Bit NF4 QLoRA Configuration & Memory Math

```python
from transformers import BitsAndBytesConfig

# 4-bit NormalFloat4 (NF4) + Double Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Native FP16 on Tesla T4
    bnb_4bit_use_double_quant=True
)
```

### Parameter & Memory Comparison:
* **Base 7B Model in FP16:** $\approx 16.0\text{ GB}$ theoretical VRAM footprint.
* **Quantized 7B Model in 4-bit NF4:** $\approx 4.5\text{ GB}$ VRAM ($\approx 72\%$ weight compression).
* **Trainable Parameters:** $20,185,088$ / $7,635,800,064$ ($0.26\%$ trainable across all linear projection layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Observed Peak Training VRAM:** $< 8.5\text{ GB}$ on a single NVIDIA Tesla T4 GPU (measured via `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`).

---

## 🔬 Evaluation & Benchmark Telemetry

### Local Hardware Benchmark (Apple Silicon M5 GPU - Experiment A):
* **First-Token Generation Latency:** $38.87\text{ ms}$ (Base) vs. $40.40\text{ ms}$ (LoRA)
* **Generation Throughput:** $35.41\text{ tokens/second}$
* **Inter-Token Latency (ITL):** $28.24\text{ ms/token}$

### 3-Tier Held-Out Test Evaluation (`test.jsonl` - 1,000 Cases):
* **Primary Metric (Diagnostic Option Match Accuracy):** Strictly parsed via bounded regular expressions (`A:`, `(B)`, `Option C`), rejecting incidental token occurrences (e.g. `'A 68-year-old...'`).
* **Secondary Metric (Completion-Only Perplexity):** Evaluated strictly on unseen physician completions (prompt tokens masked with $-100$).
* **Base Contamination Safeguard:** Base model evaluated using `with model.disable_adapter():` context manager to eliminate weight mutation or adapter leakage.

---

## 🚀 Quickstart & Execution

```bash
# 1. Clone Repository & Install Dependencies
git clone https://github.com/er3dedrw44i/clinical-llm-finetuning.git
cd clinical-llm-finetuning
pip install -r requirements.txt

# 2. Run Automated Unit Tests (Zero Data Leakage Verification)
python3 -m unittest discover tests

# 3. Materialize Persistent Splits (train.jsonl & test.jsonl)
python3 prepare_dataset.py

# 4. Run SFT Baseline Training on Mac MPS
python3 train.py

# 5. Run Evaluation on All 1,000 Held-Out Cases
python3 evaluate_model.py

# 6. Launch Interactive Clinical Decision Support Web UI
streamlit run app.py
```

---

## 📄 Defensible Resume Bullet Points

* **Parameter-Efficient Fine-Tuning:** Fine-tuned open-source LLMs on 4,000 USMLE clinical reasoning cases from MedQA using 4-bit NF4 QLoRA and LoRA adapters ($r=16, \alpha=32$, targeting `q,k,v,o,gate,up,down` layers), reducing active trainable parameters to $0.26\%$.
* **Quantization & Memory Optimization:** Applied BitsAndBytes 4-bit NF4 and double quantization to compress 7B base weights from $\approx 16\text{ GB} \rightarrow 4.5\text{ GB}$ ($\approx 72\%$ reduction), enabling single-GPU training under $8.5\text{ GB}$ peak VRAM on NVIDIA Tesla T4.
* **Completion-Only Loss Masking:** Implemented assistant-only cross-entropy loss masking (`label=-100`), preventing prompt token memorization and preserving general reasoning capabilities.
* **Evaluation & Hardware Profiling:** Evaluated base vs. fine-tuned models on a strictly held-out 1,000-case test set using diagnostic classification accuracy and completion perplexity; benchmarked Apple Silicon inference at $38.8\text{ ms}$ first-token latency and $35.4\text{ tok/s}$ throughput.

---

## 📜 License
Released under the [MIT License](LICENSE).
