"""
Step 8: Model Exporter & Weight Fusion Module.
Permanently fuses LoRA adapters into base model weights via merge_and_unload().
Produces a standalone model ready for production serving in vLLM, TGI, or Ollama.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./final_adapter"
MERGED_OUTPUT_DIR = "./merged_model"


def export_and_merge():
    print("=" * 65)
    print("      🚀 STEP 8: LoRA WEIGHT FUSION & STANDALONE EXPORT")
    print("=" * 65)

    # 1. Load Base Model
    print(f"\n1. Loading Base Model: {BASE_MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"  # Merge on CPU to preserve clean standalone weights
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, padding_side="right")

    # 2. Attach LoRA Adapter
    print(f"2. Attaching LoRA Adapter from: {ADAPTER_PATH}...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # 3. Fuse Weights: W_final = W_0 + (alpha/r) * B * A
    print("3. ⚡ Fusing weights via merge_and_unload()...")
    merged_model = peft_model.merge_and_unload()

    # 4. Save Consolidated Standalone Model
    print(f"4. 💾 Saving consolidated model to: {MERGED_OUTPUT_DIR}...")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    # 5. Summary Statistics
    total_size_mb = sum(
        os.path.getsize(os.path.join(MERGED_OUTPUT_DIR, f))
        for f in os.listdir(MERGED_OUTPUT_DIR) if os.path.isfile(os.path.join(MERGED_OUTPUT_DIR, f))
    ) / (1024 * 1024)

    print("=" * 65)
    print(f"✅ Standalone Model Successfully Exported to: {MERGED_OUTPUT_DIR}")
    print(f"  • Total Standalone Directory Size: {total_size_mb:.2f} MB")
    print(f"  • Zero-latency overhead in production!")
    print(f"  • Ready for deployment to vLLM, TGI, or GGUF conversion for Ollama!")
    print("=" * 65)


if __name__ == "__main__":
    export_and_merge()
