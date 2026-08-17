"""
Step 8: Interactive Terminal Demo.
Allows you to chat live with your fine-tuned clinical model, test medical questions,
and observe its structured clinical reasoning in real-time.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MERGED_MODEL_DIR = "./merged_model"
FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_demo_model():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_path = MERGED_MODEL_DIR if os.path.exists(MERGED_MODEL_DIR) else FALLBACK_MODEL
    print(f"📥 Loading Model from: {model_path} onto {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    ).to(device)

    return model, tokenizer, device


def generate_clinical_response(model, tokenizer, prompt, device):
    formatted = (
        "<|im_start|>system\n"
        "You are an expert Clinical Medicine AI assistant. Provide accurate, structured, and evidence-based guidance.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_chat_loop():
    print("=" * 70)
    print("      🏥 CLINICAL AI ASSISTANT: LIVE INTERACTIVE DEMO")
    print("=" * 70)

    model, tokenizer, device = load_demo_model()

    print("\n✅ System Ready! Type your medical query below (or 'quit' to exit).")
    print("-" * 70)
    print("Sample queries you can try:")
    print("  [1] A 62-year-old with CKD (eGFR 26) has diabetes. What drug is contraindicated?")
    print("  [2] Interpret ABG: pH 7.28, PaCO2 58, HCO3 26, PaO2 65.")
    print("  [3] Explain serotonin syndrome mechanism when mixing Linezolid with SSRIs.")
    print("  [4] Outline diagnostic criteria and management of acute cholecystitis.")
    print("-" * 70)

    while True:
        try:
            query = input("\n[Doctor/User] > ").strip()
            if not query or query.lower() in ["quit", "exit", "q"]:
                print("\nExiting interactive demo. Great job completing the pipeline! 👋")
                break

            if query == "1":
                query = "A 62-year-old male with chronic kidney disease (eGFR 26 mL/min) and type 2 diabetes presents for glycemic management. Which first-line agent is contraindicated, and what alternative should be recommended?"
            elif query == "2":
                query = "Interpret the following arterial blood gas (ABG) result: pH 7.28, PaCO2 58 mmHg, HCO3- 26 mEq/L, PaO2 65 mmHg on room air."
            elif query == "3":
                query = "Explain the pharmacological mechanism and risk of serotonin syndrome when co-administering Linezolid with SSRIs (e.g., Sertraline)."
            elif query == "4":
                query = "Outline the diagnostic confirmation and initial management pathway for acute calculous cholecystitis."

            print("\n🩺 [Clinical AI Response]:")
            response = generate_clinical_response(model, tokenizer, query, device)
            print(response)
            print("-" * 70)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break


if __name__ == "__main__":
    run_chat_loop()
