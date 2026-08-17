import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 1. Function to Inject LoRA Adapters into the Model
def add_lora_to_model(base_model, r=16, lora_alpha=32, lora_dropout=0.05):
    print("\n⚙️ Configuring LoRA Adapter:")
    print(f"  • Rank (r):             {r}")
    print(f"  • Alpha (scaling):      {lora_alpha}")
    print(f"  • Dropout:              {lora_dropout}")
    print(f"  • Target Modules:       All Attention & MLP Projections")

    # Define LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )

    # Injects the low-rank matrices and freezes base weights
    lora_model = get_peft_model(base_model, peft_config)
    return lora_model


# 2. Main Verification Script
if __name__ == "__main__":
    print("=" * 60)
    print("      🚀 STEP 4: LoRA ADAPTER INJECTION & VERIFICATION")
    print("=" * 60)

    # A. Detect hardware (Apple Silicon MPS vs CUDA vs CPU)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"🍎 Loading Base Model on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device)

    # B. Attach LoRA
    lora_model = add_lora_to_model(base_model, r=16, lora_alpha=32)

    # C. Print the Mathematical Proof
    print("\n--- 📊 Parameter Statistics (Proof of Efficiency) ---")
    lora_model.print_trainable_parameters()
    print("=" * 60)
    print("✅ Step 4 Complete: LoRA model is ready for training!")