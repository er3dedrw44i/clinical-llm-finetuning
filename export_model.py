"""
Step 8: Standalone 7B Weight Fusion & Exporter.
Merges LoRA adapter weights directly into the base 7B model using merge_and_unload().
Eliminates runtime adapter composition overhead.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./final_qlora_7b_adapter"
ADAPTER_WEIGHTS_FILE = os.path.join(ADAPTER_PATH, "adapter_model.safetensors")
MERGED_OUTPUT_DIR = "./merged_model_7b"


def export_merged_7b_model():
    print("=" * 70)
    print("      🚀 STEP 8: STANDALONE 7B WEIGHT FUSION & EXPORT ENGINE")
    print("=" * 70)

    if not (os.path.exists(ADAPTER_WEIGHTS_FILE) and os.path.getsize(ADAPTER_WEIGHTS_FILE) > 1000):
        raise FileNotFoundError(
            f"❌ Trained adapter weights not found at {ADAPTER_WEIGHTS_FILE}.\n"
            "Please complete 7B QLoRA training in Google Colab (qlora_colab.ipynb) or on an NVIDIA GPU, "
            "and place `adapter_model.safetensors` in `final_qlora_7b_adapter/` before exporting merged weights."
        )

    print(f"\n1. Loading Base 7B Model in FP16 for clean weight fusion...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    print(f"2. Attaching trained QLoRA adapter from {ADAPTER_PATH}...")
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("3. Fusing adapter weights into base model via merge_and_unload()...")
    merged_model = lora_model.merge_and_unload()

    print(f"4. Saving standalone consolidated weights to: {MERGED_OUTPUT_DIR}...")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    print("=" * 70)
    print(f"🎉 7B Standalone Merged Model successfully exported to: {MERGED_OUTPUT_DIR}")
    print("✅ Eliminates runtime LoRA adapter composition overhead.")
    print("=" * 70)


if __name__ == "__main__":
    export_merged_7b_model()
