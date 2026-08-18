"""
Step 6: Production 0.5B Model Evaluation Engine (Apple Silicon MPS / Local).
Evaluates Base Qwen2.5-0.5B vs. Fine-Tuned (LoRA) Model strictly on `test.jsonl` (1,000 cases).
Features:
1. Exact same safe tokenization and prompt truncation policy as training.
2. Eliminates adapter contamination using `with model.disable_adapter():`.
3. Strict regex option parsing rejecting false positive article 'A'.
4. Primary metric: Diagnostic Option Match Accuracy (%) with difference in percentage points (pp).
"""

import os
import re
import math
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from prepare_dataset import load_split

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./final_adapter"
TEST_DATA_PATH = "test.jsonl"


def extract_predicted_option(text: str) -> str:
    """Strictly extracts option letter (A, B, C, D, E). Rejects 'A 68-year-old...'."""
    if not text:
        return "NONE"
    clean = text.strip()
    
    # Pattern 1: Leading option with delimiter (A:, A., A), A -, (A))
    m1 = re.match(r'^\s*\(?([A-Ea-e])\)?\s*[:\.\)\-]\s*', clean)
    if m1:
        return m1.group(1).upper()

    # Pattern 2: Explicit answer phrasing (Option A, Answer: A, The correct answer is B)
    m2 = re.search(r'(?:option|answer(?:\s*is)?)\s*[:\s\-]*\(?([A-Ea-e])\)?(?:\b|[\.\:\)\-])', clean, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    # Pattern 3: Exact standalone 1-token output
    tokens = clean.split()
    if len(tokens) == 1:
        tok = tokens[0].upper().rstrip('.:,)')
        if tok in ["A", "B", "C", "D", "E"]:
            return tok

    return "NONE"


def format_eval_tokens_safely(prompt_str, output_str, tokenizer, max_length=512):
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    output_ids = tokenizer.encode(output_str + "<|im_end|>", add_special_tokens=False)

    if len(output_ids) >= max_length:
        output_ids = output_ids[:max_length - 1]

    if len(prompt_ids) + len(output_ids) > max_length:
        max_prompt_len = max(1, max_length - len(output_ids))
        prompt_ids = prompt_ids[-max_prompt_len:]

    full_ids = prompt_ids + output_ids
    labels = [-100] * len(prompt_ids) + output_ids
    return full_ids, labels, len(prompt_ids)


def compute_completion_perplexity(model, tokenizer, test_samples, device, max_length=512):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for sample in test_samples:
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            output = sample["output"]
            context_str = f"\nContext: {input_text}" if input_text else ""

            prompt_str = (
                "<|im_start|>system\n"
                "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
                f"<|im_start|>user\n{instruction}{context_str}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            full_ids, labels, prompt_len = format_eval_tokens_safely(prompt_str, output, tokenizer, max_length=max_length)

            if len(full_ids) <= prompt_len:
                continue

            input_tensor = torch.tensor([full_ids], device=device)
            logits = model(input_tensor).logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_tensor[..., 1:].contiguous()

            completion_start = max(0, prompt_len - 1)
            comp_logits = shift_logits[:, completion_start:, :].view(-1, shift_logits.size(-1))
            comp_labels = shift_labels[:, completion_start:].view(-1)

            if comp_labels.numel() > 0:
                loss = F.cross_entropy(comp_logits, comp_labels)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    losses.append(loss.item())

    if not losses:
        return 1.0
    mean_loss = sum(losses) / len(losses)
    return round(math.exp(min(20.0, mean_loss)), 2)


def generate_response(model, tokenizer, instruction, input_text, device):
    context_str = f"\nContext: {input_text}" if input_text else ""
    prompt = (
        "<|im_start|>system\n"
        "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}{context_str}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_evaluation(num_test_eval=None):
    print("=" * 70)
    print("      🚀 STEP 6: 0.5B BASE VS. LORA EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)

    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🍎 Active Hardware Device: {device}")

    all_test_samples = load_split(TEST_DATA_PATH)
    test_samples = all_test_samples if num_test_eval is None else all_test_samples[:num_test_eval]
    print(f"  • Loaded strictly held-out test split: {TEST_DATA_PATH}")
    print(f"  • Total test set size: {len(all_test_samples)} cases | Evaluating on: {len(test_samples)} cases.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # 1. Primary Metric: Strict Diagnostic Option Match Accuracy
    print("\n--- 🎯 PRIMARY METRIC: STRICT DIAGNOSTIC OPTION MATCH ACCURACY (%) ---")
    base_matches = 0
    tuned_matches = 0

    for idx, sample in enumerate(test_samples, 1):
        inst = sample["instruction"]
        inp = sample.get("input", "")
        gold = sample["output"].strip()
        gold_opt = extract_predicted_option(gold)

        with tuned_model.disable_adapter():
            base_resp = generate_response(tuned_model, tokenizer, inst, inp, device)
        
        tuned_resp = generate_response(tuned_model, tokenizer, inst, inp, device)

        base_pred_opt = extract_predicted_option(base_resp)
        tuned_pred_opt = extract_predicted_option(tuned_resp)

        if base_pred_opt != "NONE" and base_pred_opt == gold_opt:
            base_matches += 1
        if tuned_pred_opt != "NONE" and tuned_pred_opt == gold_opt:
            tuned_matches += 1

        if idx <= 5 or idx == len(test_samples):
            print(f"  [Case {idx:04d}] Gold: {gold_opt} | Base Pred: {base_pred_opt:<4} | LoRA Pred: {tuned_pred_opt:<4}")

    base_acc = round((base_matches / len(test_samples)) * 100, 2)
    tuned_acc = round((tuned_matches / len(test_samples)) * 100, 2)
    diff_pp = round(tuned_acc - base_acc, 2)

    # 2. Secondary Metric: Completion Perplexity (Exact Training Alignment)
    print("\n--- 📊 SECONDARY METRIC: COMPLETION PERPLEXITY (PPL) ---")
    with tuned_model.disable_adapter():
        base_ppl = compute_completion_perplexity(tuned_model, tokenizer, test_samples, device, max_length=512)
    tuned_ppl = compute_completion_perplexity(tuned_model, tokenizer, test_samples, device, max_length=512)

    # Report with percentage points (pp)
    print("\n" + "=" * 70)
    print("                     0.5B BENCHMARK EVALUATION REPORT")
    print("=" * 70)
    print(f"  Metric                     Base 0.5B         Fine-Tuned (LoRA)  Difference")
    print(f"  -----------------------------------------------------------------------")
    print(f"  Diagnostic Accuracy (%)    {base_acc:<17}% {tuned_acc:<18}% {'+' if diff_pp >= 0 else ''}{diff_pp} pp")
    print(f"  Completion Perplexity      {base_ppl:<17} {tuned_ppl:<18} {'-' if tuned_ppl <= base_ppl else '+'}{abs(base_ppl - tuned_ppl):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
