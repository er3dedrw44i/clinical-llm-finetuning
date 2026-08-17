"""
Step 3: Hardware-Aware Model & Tokenizer Loader.
Automatically routes between:
- Experiment A (Mac Apple Silicon MPS): FP32/BF16 lightweight model for local debugging.
- Experiment B (NVIDIA CUDA): BitsAndBytes 4-bit NF4 Quantization + Double Quantization for 7B/8B QLoRA.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

LOCAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CUDA_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # or meta-llama/Meta-Llama-3-8B-Instruct


def get_bnb_4bit_config():
    """Returns production BitsAndBytes 4-bit NF4 Quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",               # Information-theoretically optimal for Gaussian weights
        bnb_4bit_compute_dtype=torch.bfloat16,   # Dequantize to BF16 for matrix operations
        bnb_4bit_use_double_quant=True           # Double quantization saves ~0.4GB additional VRAM
    )


def detect_hardware():
    if torch.cuda.is_available():
        print("🚀 NVIDIA GPU detected! Selecting Experiment B: 4-bit NF4 QLoRA Pipeline on CUDA.")
        return "cuda", CUDA_MODEL_NAME
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🍎 Apple Silicon GPU detected! Selecting Experiment A: Development LoRA Pipeline on MPS.")
        return "mps", LOCAL_MODEL_NAME
    else:
        print("💻 CPU detected! Running in CPU fallback mode.")
        return "cpu", LOCAL_MODEL_NAME


def load_tokenizer(model_name):
    print(f"📥 Loading Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name, device):
    """
    Loads base model with automatic 4-bit NF4 quantization on CUDA,
    or FP32 on Apple Silicon MPS / CPU.
    """
    if device == "cuda":
        print(f"⚡ Applying BitsAndBytes 4-bit NF4 Quantization to: {model_name}...")
        bnb_config = get_bnb_4bit_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        # Prepare 4-bit weights for LoRA gradient checkpointing
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"📥 Loading Base Model: {model_name} (torch.float32 on {device})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)

    return model


if __name__ == "__main__":
    print("=" * 65)
    print("      🚀 STEP 3: UNIFIED HARDWARE-AWARE MODEL LOADER")
    print("=" * 65)

    device, model_name = detect_hardware()
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, device)

    print("\n✅ Base Model Successfully Loaded!")
    print(f"  • Active Device: {device}")
    print(f"  • Selected Model: {model_name}")
    print(f"  • 4-Bit NF4 Quantization: {'ACTIVE (BitsAndBytes)' if device == 'cuda' else 'INACTIVE (Running on Apple Silicon MPS)'}")
    print("=" * 65)
