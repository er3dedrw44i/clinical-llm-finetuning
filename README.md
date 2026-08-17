# Domain-Specific Clinical LLM Fine-Tuning & Evaluation Pipeline

A production-grade, dual-track LLM fine-tuning, evaluation, and deployment pipeline for clinical decision support. Built with **PyTorch**, **Hugging Face Transformers**, **PEFT / LoRA**, **BitsAndBytes (4-bit NF4 QLoRA)**, and **Weights & Biases (W&B)**.

---

## 🎯 Dual-Experiment Engineering Architecture

To guarantee both fast local development and single-GPU production scalability, the pipeline is structured into two complementary tracks:

```
                            DUAL-EXPERIMENT ARCHITECTURE
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
  [ EXPERIMENT A: Local Development ]                [ EXPERIMENT B: Production QLoRA ]
  • Hardware : Apple Silicon Metal (MPS), 16GB RAM   • Hardware : NVIDIA Cloud GPU (CUDA)
  • Model    : Qwen2.5-0.5B / 1.5B                   • Model    : Qwen2.5-7B / Llama-3-8B
  • Tech     : FP32 + LoRA (PEFT r=16, alpha=32)     • Tech     : BitsAndBytes 4-bit NF4 + Double Quant
  • Focus    : Latency benchmarks, unit tests,       • Focus    : VRAM memory optimization benchmark
               and interactive Streamlit web demo.                (140 GB -> 8.5 GB footprint).
```

---

## 📊 Quantization & Memory Optimization Benchmark (8B Model Footprint)

| Fine-Tuning Technique | Precision | Model Weights | AdamW Optimizer States | Total VRAM Required | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | FP16 / BF16 | $16.0\text{ GB}$ | $64.0\text{ GB}$ ($16\text{ bytes/param}$) | **$\approx 128 - 140\text{ GB}$** | Multi-GPU Cluster ($2\times \text{A100 } 80\text{GB}$) |
| **Standard LoRA** | 16-bit Base + FP32 Adapters | $16.0\text{ GB}$ (Frozen) | $\approx 1.5\text{ GB}$ | **$\approx 20 - 24\text{ GB}$** | $1\times \text{RTX 3090 / A10G}$ |
| **4-Bit NF4 QLoRA** | **4-bit NF4 + Double Quant** | **$4.5\text{ GB}$** | **$\approx 1.2\text{ GB}$** | **$\approx 6.0 - 8.5\text{ GB}$** | **Single GPU ($1\times \text{RTX 4060 / T4}$)** |

> **Key Takeaway:** BitsAndBytes 4-bit NF4 quantization reduces base weight memory by **$72\%$**, and PEFT reduces trainable parameters by **$>98\%$** ($1.75\%$ trainable parameters), enabling single-GPU execution of 7B–8B models.

---

## 🏗️ The 8-Step Pipeline Modules

```
  1. [DATASET]           ──► 5,000 USMLE clinical reasoning cases (4,000 train / 1,000 test)
  2. [DATA PREPARATION]  ──► ETL cleaning, context-aware deduplication & fixed-seed 80/20 split
  3. [MODEL LOADER]      ──► Hardware-aware routing (MPS FP32 / CUDA 4-bit NF4 BitsAndBytes)
  4. [PEFT / LoRA]       ──► Low-Rank Adapter injection (r=16, alpha=32, target: ALL linear layers)
  5. [SFT TRAINING]      ──► TRL SFTTrainer + completion-only loss masking + W&B tracking
  6. [3-TIER EVALUATION] ──► Completion PPL, Multiset Counter Token F1 & LLM-as-a-Judge rubrics
  7. [BENCHMARKING]      ──► Hardware profiling: TTFT (38ms), Throughput (35.4 tok/s), ITL (28ms)
  8. [EXPORT & SERVING]  ──► merge_and_unload() weight fusion + interactive Streamlit web demo
```

---

## ⚡ Inference Latency & Hardware Benchmarks (Apple Silicon M5 MPS)

* **Time-to-First-Token (TTFT):** `38.87 ms` (Base) vs `40.40 ms` (LoRA) — *LoRA overhead: $+1.53\text{ ms}$*
* **Generation Throughput:** `35.41 tokens/second` ($>4\times$ human reading speed).
* **Inter-Token Latency (ITL):** `28.24 ms/token`.
* **Zero-Latency Serving:** Applied `merge_and_unload()` to permanently fuse $W_{\text{final}} = W_0 + \frac{\alpha}{r}(B \cdot A)$, eliminating dynamic adapter overhead for deployment in **vLLM**, **TGI**, and **Ollama (GGUF)**.

---

## 🚀 Quickstart & Pipeline Execution

### Local Mac M5 Execution (Experiment A):
```bash
# 1. Prepare domain dataset
python3 prepare_dataset.py

# 2. Hardware auto-detection & model load
python3 model_loader.py

# 3. Inject LoRA adapters & verify parameter reduction
python3 lora_setup.py

# 4. SFT training loop with W&B logging
python3 train.py

# 5. 3-Tier scientific evaluation harness
python3 evaluate_model.py

# 6. Benchmark latency, TTFT, and throughput
python3 benchmark.py

# 7. Fuse weights and export standalone model
python3 export_model.py

# 8. Launch Streamlit Web UI
streamlit run app.py
```

### Production 4-bit QLoRA on NVIDIA GPU (Experiment B):
Open and run **`qlora_colab.ipynb`** in Google Colab (with a free T4 GPU runtime) or run:
```bash
python3 qlora_train.py
```
