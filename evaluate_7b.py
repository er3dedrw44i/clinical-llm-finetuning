"""
Step 6B: Production 7B Model Evaluation Engine (CUDA / Cloud).
Evaluates Base Qwen2.5-7B-Instruct vs. Fine-Tuned QLoRA 7B strictly on `test.jsonl` (1,000 cases).
Prioritizes Diagnostic Option Match Accuracy (%) using strict regex parsing.
"""

import os
import re
import json
import math
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from prepare_dataset import load_split

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./final_qlora_7b_adapter"
TEST_DATA_PATH = "test.jsonl"


def extract_predicted_option(text: str) -> str:
    """
    Strictly extracts option letter (A, B, C, D, E) from model generation.
    Rejects accidental substring occurrences (e.g., 'b' in 'Staphylococcus').
    """
    if not text:
        return "NONE"
    clean_text = text.strip()
    
    # Pattern 1: Leading option letter (e.g. 'A: ...', 'B. ...', '(C)', 'D - ...')
    m1 = re.match(r'^\s*\(?([A-Ea-e])\)?(?:\s*[:\.\)\-]|\s+|$)', clean_text)
    if m1:
        return m1.group(1).upper()

    # Pattern 2: Explicit answer phrasing (e.g. 'The correct answer is B', 'Option C')
    m2 = re.search(r'(?:option|answer(?:\s*is)?)\s*[:\s\-]*\(?([A-Ea-e])\)?(?:\b|[\.\:\)])', clean_text, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    # Pattern 3: Standalone first token is a valid option letter
    first_tok = clean_text.split()[0].upper().rstrip('.:,)') if clean_text.split() else ""
    if first_tok in ["A", "B", "C", "D", "E"]:
        return first_tok

    return "NONE"


def get_bnb_4bit_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


def compute_completion_perplexity(model, tokenizer, test_samples, device):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for sample in test_samples:
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            output = sample["output"]
            context_str = f"\nContext: {input_text}" if input_text else ""

            prompt_text = (
                "<|im_start|>system\n"
                "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
                f"<|im_start|>user\n{instruction}{context_str}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            full_text = prompt_text + f"{output}<|im_end|>"

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            if len(full_ids) <= len(prompt_ids):
                continue

            input_tensor = torch.tensor([full_ids], device=device)
            logits = model(input_tensor).logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_tensor[..., 1:].contiguous()

            completion_start = max(0, len(prompt_ids) - 1)
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


def run_7b_evaluation(num_test_eval=None):
    print("=" * 70)
    print("      🚀 STEP 6B: 7B BASE VS. QLORA EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Active Hardware: {device.upper()}")

    all_test_samples = load_split(TEST_DATA_PATH)
    test_samples = all_test_samples if num_test_eval is None else all_test_samples[:num_test_eval]
    print(f"  • Loaded strictly held-out test split: {TEST_DATA_PATH}")
    print(f"  • Total test set size: {len(all_test_samples)} cases | Evaluating on: {len(test_samples)} cases.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load 4-Bit Base 7B Model
    bnb_config = get_bnb_4bit_config() if device == "cuda" else None
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        base_model = base_model.to(device)

    # Attach QLoRA Adapter if available
    tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) if os.path.exists(ADAPTER_PATH) else base_model

    # 1. Primary Metric: Strict Diagnostic Option Match Accuracy
    print("\n--- 🎯 PRIMARY METRIC: STRICT DIAGNOSTIC OPTION MATCH ACCURACY (%) ---")
    base_matches = 0
    tuned_matches = 0

    for idx, sample in enumerate(test_samples, 1):
        inst = sample["instruction"]
        inp = sample.get("input", "")
        gold = sample["output"].strip()
        gold_opt = extract_predicted_option(gold)

        base_resp = generate_response(base_model, tokenizer, inst, inp, device)
        tuned_resp = generate_response(tuned_model, tokenizer, inst, inp, device)

        base_pred_opt = extract_predicted_option(base_resp)
        tuned_pred_opt = extract_predicted_option(tuned_resp)

        if base_pred_opt != "NONE" and base_pred_opt == gold_opt:
            base_matches += 1
        if tuned_pred_opt != "NONE" and tuned_pred_opt == gold_opt:
            tuned_matches += 1

        if idx <= 5 or idx == len(test_samples):
            print(f"  [Case {idx:04d}] Gold: {gold_opt} | Base Pred: {base_pred_opt:<4} | QLoRA Pred: {tuned_pred_opt:<4}")

    base_acc = round((base_matches / len(test_samples)) * 100, 2)
    tuned_acc = round((tuned_matches / len(test_samples)) * 100, 2)

    # 2. Secondary Metric: Completion Perplexity
    print("\n--- 📊 SECONDARY METRIC: COMPLETION PERPLEXITY (PPL) ---")
    base_ppl = compute_completion_perplexity(base_model, tokenizer, test_samples, device)
    tuned_ppl = compute_completion_perplexity(tuned_model, tokenizer, test_samples, device)

    # Dynamic Benchmark Report
    print("\n" + "=" * 70)
    print("                     7B BENCHMARK EVALUATION REPORT")
    print("=" * 70)
    print(f"  Metric                     Base 7B           QLoRA 7B           Difference")
    print(f"  -----------------------------------------------------------------------")
    print(f"  Diagnostic Accuracy (%)    {base_acc:<17}% {tuned_acc:<18}% {'+' if tuned_acc >= base_acc else '-'}{abs(tuned_acc - base_acc):.2f}%")
    print(f"  Completion Perplexity      {base_ppl:<17} {tuned_ppl:<18} {'-' if tuned_ppl <= base_ppl else '+'}{abs(base_ppl - tuned_ppl):.2f}")
    print("=" * 70)

    if tuned_acc > base_acc or (tuned_acc == base_acc and tuned_ppl < base_ppl):
        print("🏆 BENCHMARK RESULT: Fine-Tuned QLoRA 7B Model demonstrated higher empirical diagnostic performance.")
    elif base_acc > tuned_acc:
        print("🏆 BENCHMARK RESULT: Base 7B Model baseline outperformed fine-tuned model.")
    else:
        print("⚖️ BENCHMARK RESULT: Base 7B and QLoRA 7B demonstrated identical diagnostic accuracy.")
    print("=" * 70)


if __name__ == "__main__":
    run_7b_evaluation()
