"""
Step 7: Inference Latency & Throughput Benchmarking Engine.
Measures:
1. Time-to-First-Token (TTFT in milliseconds).
2. Generation Throughput (tokens per second).
3. Inter-Token Latency (ITL in ms per token).
4. Base Model vs. Fine-Tuned (LoRA) Comparative Performance.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./final_adapter"
BENCHMARK_PROMPT = "Explain the clinical management of acute calculous cholecystitis."


def measure_inference_latency(model, tokenizer, prompt, device, max_new_tokens=64, num_trials=3):
    """
    Benchmarks TTFT, Throughput (tokens/sec), and ITL across multiple trials.
    Includes 2 warm-up passes to eliminate GPU compilation overhead.
    """
    model.eval()
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    # 1. Warm-up passes
    for _ in range(2):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    ttft_trials = []
    throughput_trials = []
    total_time_trials = []
    token_counts = []

    # 2. Benchmark Trials
    for _ in range(num_trials):
        # A. Measure TTFT (First Token)
        start_ttft = time.perf_counter()
        with torch.no_grad():
            first_out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        ttft_ms = (time.perf_counter() - start_ttft) * 1000.0
        ttft_trials.append(ttft_ms)

        # B. Measure Full Generation
        start_full = time.perf_counter()
        with torch.no_grad():
            full_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        duration = time.perf_counter() - start_full
        
        num_generated = full_out.shape[1] - input_length
        token_counts.append(num_generated)
        total_time_trials.append(duration)

        if duration > 0 and num_generated > 0:
            tok_per_sec = num_generated / duration
            throughput_trials.append(tok_per_sec)

    avg_ttft = sum(ttft_trials) / len(ttft_trials)
    avg_throughput = sum(throughput_trials) / len(throughput_trials) if throughput_trials else 0.0
    avg_duration = sum(total_time_trials) / len(total_time_trials)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_itl = (1000.0 / avg_throughput) if avg_throughput > 0 else 0.0

    return {
        "ttft_ms": round(avg_ttft, 2),
        "throughput_tok_sec": round(avg_throughput, 2),
        "itl_ms_per_token": round(avg_itl, 2),
        "duration_sec": round(avg_duration, 3),
        "tokens_generated": round(avg_tokens, 1)
    }


def run_benchmark():
    print("=" * 70)
    print("      🚀 STEP 7: INFERENCE LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 70)

    # 1. Detect Device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"🍎 Active Hardware Accelerator: {device}")

    # 2. Load Tokenizer & Base Model
    print("\n1. Loading Base Foundation Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device)

    # 3. Load Fine-Tuned (LoRA) Model
    print("2. Loading Fine-Tuned (LoRA) Model...")
    tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # 4. Run Benchmarks
    print("\n3. ⚡ Benchmarking Base Model (3 trials + warmups)...")
    base_metrics = measure_inference_latency(base_model, tokenizer, BENCHMARK_PROMPT, device)

    print("4. ⚡ Benchmarking Fine-Tuned (LoRA) Model (3 trials + warmups)...")
    tuned_metrics = measure_inference_latency(tuned_model, tokenizer, BENCHMARK_PROMPT, device)

    # 5. Display Benchmark Report
    print("\n" + "=" * 70)
    print("                LATENCY & THROUGHPUT BENCHMARK REPORT")
    print("=" * 70)
    print(f"  Benchmark Metric               Base Model        Fine-Tuned (LoRA)  Overhead")
    print(f"  -------------------------------------------------------------------------")
    print(f"  Time-to-First-Token (TTFT)     {base_metrics['ttft_ms']:<6} ms         {tuned_metrics['ttft_ms']:<6} ms         {tuned_metrics['ttft_ms'] - base_metrics['ttft_ms']:+.2f} ms")
    print(f"  Throughput (Tokens / Sec)      {base_metrics['throughput_tok_sec']:<6} tok/s      {tuned_metrics['throughput_tok_sec']:<6} tok/s      {tuned_metrics['throughput_tok_sec'] - base_metrics['throughput_tok_sec']:+.2f} tok/s")
    print(f"  Inter-Token Latency (ITL)      {base_metrics['itl_ms_per_token']:<6} ms/tok     {tuned_metrics['itl_ms_per_token']:<6} ms/tok     {tuned_metrics['itl_ms_per_token'] - base_metrics['itl_ms_per_token']:+.2f} ms")
    print(f"  Total Duration ({base_metrics['tokens_generated']} tokens)      {base_metrics['duration_sec']:<6} s          {tuned_metrics['duration_sec']:<6} s          {tuned_metrics['duration_sec'] - base_metrics['duration_sec']:+.3f} s")
    print("=" * 70)
    print(f"✅ Benchmark Complete on Apple Silicon ({device})!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
