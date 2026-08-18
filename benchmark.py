"""
Step 7: Production 7B Latency & Throughput Benchmark Engine.
Measures:
1. First-Token Generation Latency (ms)
2. Generation Throughput (tokens/second)
3. Inter-Token Latency (ITL)
4. Peak VRAM Footprint
"""

import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./final_qlora_7b_adapter"
ADAPTER_WEIGHTS_FILE = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")


def benchmark_7b_inference(num_warmup=2, num_runs=5):
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Hardware Accelerator: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    ) if is_cuda else None

    print(f"📥 Loading {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype if is_cuda else torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" if is_cuda else None
    )
    if not is_cuda:
        base_model = base_model.to(device)

    # Check for real adapter weights
    has_weights = os.path.exists(ADAPTER_WEIGHTS_FILE) and os.path.getsize(ADAPTER_WEIGHTS_FILE) > 1000
    if has_weights:
        print(f"✅ Loading fine-tuned adapter weights from {ADAPTER_WEIGHTS_FILE}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print("💡 No fine-tuned adapter weights detected; benchmarking Base 7B Model.")
        model = base_model

    model.eval()

    prompt = (
        "<|im_start|>system\n"
        "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
        "<|im_start|>user\nA 68-year-old male presents with fever, rust-colored sputum, and dyspnea. What is the diagnosis?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    print("⚡ Warming up GPU kernels...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        if is_cuda: torch.cuda.synchronize()

    # Benchmark First-Token Generation Latency
    print(f"⏱️ Measuring First-Token Latency over {num_runs} runs...")
    latencies = []
    for _ in range(num_runs):
        if is_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        if is_cuda: torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_first_token_ms = round(sum(latencies) / len(latencies), 2)

    # Benchmark Throughput
    print(f"🚀 Measuring Generation Throughput (64 tokens)...")
    if is_cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    if is_cuda: torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    throughput_tps = round(gen_tokens / total_time, 2)
    itl_ms = round((total_time / gen_tokens) * 1000, 2)

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else 0.0

    print("=" * 70)
    print("                     7B INFERENCE BENCHMARK REPORT")
    print("=" * 70)
    print(f"  • Model:                        {MODEL_NAME}")
    print(f"  • Fine-Tuned Adapter Loaded:    {has_weights}")
    print(f"  • Hardware:                     {device.upper()}")
    print(f"  • First-Token Latency:          {mean_first_token_ms} ms")
    print(f"  • Generation Throughput:        {throughput_tps} tokens/second")
    print(f"  • Inter-Token Latency (ITL):    {itl_ms} ms/token")
    if is_cuda:
        print(f"  • Peak Inference VRAM:          {peak_vram:.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_7b_inference()
