"""
Interactive Terminal Demo for 7B QLoRA Clinical AI Model.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from data_utils import extract_predicted_option

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./final_qlora_7b_adapter"


def run_interactive_demo():
    print("=" * 70)
    print("      🩺 7B QLoRA CLINICAL DECISION SUPPORT INTERACTIVE TERMINAL")
    print("=" * 70)

    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

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

    print(f"📥 Loading {MODEL_NAME} on {device.upper()}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype if is_cuda else torch.float32,
        device_map="auto" if is_cuda else None
    )
    if not is_cuda:
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH) if os.path.exists(ADAPTER_PATH) else base_model
    model.eval()

    print("✅ Model loaded! Enter your USMLE clinical case vignette (or 'exit' to quit):\n")

    while True:
        try:
            user_input = input("🩺 Enter Clinical Vignette & Options:\n> ")
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                print("Exiting demo. Goodbye!")
                break
            if not user_input.strip():
                continue

            prompt = (
                "<|im_start|>system\n"
                "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.2,
                    do_sample=False,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id
                )

            resp = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            parsed_opt = extract_predicted_option(resp)

            print("\n" + "=" * 50)
            print(f"🩺 [7B Clinical Response]:\n{resp}")
            if parsed_opt != "NONE":
                print(f"🎯 [Parsed Diagnosis]: Option {parsed_opt}")
            print("=" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nExiting demo.")
            break


if __name__ == "__main__":
    run_interactive_demo()
