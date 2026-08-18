"""
Step 6B: Production 7B Model Evaluation Engine (CUDA / Cloud).
Evaluates Base Qwen2.5-7B-Instruct vs. Fine-Tuned QLoRA 7B strictly on `test.jsonl` (1,000 cases).
Features:
1. Shared preprocessing via `data_utils.py`.
2. Eliminates adapter contamination using `with model.disable_adapter():`.
3. Computes 95% Confidence Intervals & McNemar paired transition matrix.
4. Automatically exports `results/qlora_7b_evaluation.json` and `results/error_analysis.json`.
"""

import os
import math
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from data_utils import (
    load_split,
    extract_predicted_option,
    format_single_example_tokens,
    DEFAULT_MAX_LENGTH
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./final_qlora_7b_adapter"
TEST_DATA_PATH = "test.jsonl"
RESULTS_DIR = "./results"


def get_bnb_4bit_config():
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )


def compute_completion_perplexity(model, tokenizer, test_samples, device, max_length=DEFAULT_MAX_LENGTH):
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
            full_ids, labels, prompt_len = format_single_example_tokens(
                prompt_str, output, tokenizer, max_length=max_length
            )

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


def calculate_ci_95(correct: int, total: int) -> Tuple[float, float]:
    """Computes Wilson 95% Confidence Interval for binomial accuracy."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    z = 1.96
    denominator = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total) / denominator
    return (round(max(0.0, center - spread) * 100, 2), round(min(1.0, center + spread) * 100, 2))


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

    bnb_config = get_bnb_4bit_config() if device == "cuda" else None
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device != "cuda":
        base_model = base_model.to(device)

    tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) if os.path.exists(ADAPTER_PATH) else base_model

    # 1. Primary Metric & Error Analysis
    print("\n--- 🎯 PRIMARY METRIC & PAIRED TRANSITION EVALUATION ---")
    base_matches, tuned_matches = 0, 0
    transitions = {
        "wrong_to_correct": 0,
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_wrong": 0
    }
    option_accuracy = {opt: {"base_correct": 0, "tuned_correct": 0, "total": 0} for opt in ["A", "B", "C", "D", "E"]}

    for idx, sample in enumerate(test_samples, 1):
        inst = sample["instruction"]
        inp = sample.get("input", "")
        gold = sample["output"].strip()
        gold_opt = extract_predicted_option(gold)

        if hasattr(tuned_model, "disable_adapter"):
            with tuned_model.disable_adapter():
                base_resp = generate_response(tuned_model, tokenizer, inst, inp, device)
        else:
            base_resp = generate_response(base_model, tokenizer, inst, inp, device)

        tuned_resp = generate_response(tuned_model, tokenizer, inst, inp, device)

        base_pred_opt = extract_predicted_option(base_resp)
        tuned_pred_opt = extract_predicted_option(tuned_resp)

        base_is_correct = (base_pred_opt != "NONE" and base_pred_opt == gold_opt)
        tuned_is_correct = (tuned_pred_opt != "NONE" and tuned_pred_opt == gold_opt)

        if base_is_correct: base_matches += 1
        if tuned_is_correct: tuned_matches += 1

        # Track transitions
        if not base_is_correct and tuned_is_correct:
            transitions["wrong_to_correct"] += 1
        elif base_is_correct and tuned_is_correct:
            transitions["correct_to_correct"] += 1
        elif base_is_correct and not tuned_is_correct:
            transitions["correct_to_wrong"] += 1
        else:
            transitions["wrong_to_wrong"] += 1

        # Track by option
        if gold_opt in option_accuracy:
            option_accuracy[gold_opt]["total"] += 1
            if base_is_correct: option_accuracy[gold_opt]["base_correct"] += 1
            if tuned_is_correct: option_accuracy[gold_opt]["tuned_correct"] += 1

        if idx <= 5 or idx == len(test_samples):
            print(f"  [Case {idx:04d}] Gold: {gold_opt} | Base: {base_pred_opt:<4} | QLoRA: {tuned_pred_opt:<4}")

    base_acc = round((base_matches / len(test_samples)) * 100, 2)
    tuned_acc = round((tuned_matches / len(test_samples)) * 100, 2)
    diff_pp = round(tuned_acc - base_acc, 2)

    base_ci = calculate_ci_95(base_matches, len(test_samples))
    tuned_ci = calculate_ci_95(tuned_matches, len(test_samples))

    # 2. Secondary Metric: Completion Perplexity
    print("\n--- 📊 SECONDARY METRIC: COMPLETION PERPLEXITY (PPL) ---")
    if hasattr(tuned_model, "disable_adapter"):
        with tuned_model.disable_adapter():
            base_ppl = compute_completion_perplexity(tuned_model, tokenizer, test_samples, device)
    else:
        base_ppl = compute_completion_perplexity(base_model, tokenizer, test_samples, device)
    tuned_ppl = compute_completion_perplexity(tuned_model, tokenizer, test_samples, device)

    # Automatically save result artifacts
    os.makedirs(RESULTS_DIR, exist_ok=True)
    eval_artifact = {
        "model_evaluated": MODEL_NAME,
        "adapter_path": ADAPTER_PATH,
        "eval_dataset": f"test.jsonl ({len(test_samples)} cases)",
        "quantization": "BitsAndBytes 4-bit NF4",
        "primary_metric": "Strict Diagnostic Option Match Accuracy (%)",
        "base_7b_accuracy_pct": base_acc,
        "base_7b_95_ci_pct": base_ci,
        "qlora_7b_accuracy_pct": tuned_acc,
        "qlora_7b_95_ci_pct": tuned_ci,
        "accuracy_gain_percentage_points_pp": diff_pp,
        "base_7b_completion_ppl": base_ppl,
        "qlora_7b_completion_ppl": tuned_ppl,
        "base_contamination_safeguard": "with model.disable_adapter(): context manager",
        "tokenization_policy": f"Shared data_utils.py formatting (max_length={DEFAULT_MAX_LENGTH})"
    }
    with open(os.path.join(RESULTS_DIR, "qlora_7b_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(eval_artifact, f, indent=2)

    error_artifact = {
        "total_evaluated": len(test_samples),
        "paired_transitions": transitions,
        "breakdown_by_option": option_accuracy,
        "key_finding": f"QLoRA resolved {transitions['wrong_to_correct']} baseline errors while preserving {transitions['correct_to_correct']} correct baseline diagnoses."
    }
    with open(os.path.join(RESULTS_DIR, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(error_artifact, f, indent=2)

    print("\n" + "=" * 70)
    print("                     7B BENCHMARK EVALUATION REPORT")
    print("=" * 70)
    print(f"  Metric                     Base 7B           QLoRA 7B           Difference")
    print(f"  -----------------------------------------------------------------------")
    print(f"  Diagnostic Accuracy (%)    {base_acc:<7}% [{base_ci[0]}-{base_ci[1]}]  {tuned_acc:<7}% [{tuned_ci[0]}-{tuned_ci[1]}]  {'+' if diff_pp >= 0 else ''}{diff_pp} pp")
    print(f"  Completion Perplexity      {base_ppl:<17} {tuned_ppl:<18} {'-' if tuned_ppl <= base_ppl else '+'}{abs(base_ppl - tuned_ppl):.2f}")
    print("=" * 70)
    print(f"  📊 Paired Error Transitions: Wrong➔Correct: {transitions['wrong_to_correct']} | Correct➔Correct: {transitions['correct_to_correct']} | Correct➔Wrong: {transitions['correct_to_wrong']} | Wrong➔Wrong: {transitions['wrong_to_wrong']}")
    print(f"  💾 Results automatically saved to: results/qlora_7b_evaluation.json and results/error_analysis.json")
    print("=" * 70)


if __name__ == "__main__":
    run_7b_evaluation()
