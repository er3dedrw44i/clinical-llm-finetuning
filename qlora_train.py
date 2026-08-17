"""
Production 4-Bit NF4 QLoRA Training Engine (NVIDIA CUDA).
Integrates:
1. Full domain dataset (medical_domain_dataset.jsonl) with 80/20 train/test split.
2. BitsAndBytes 4-bit NormalFloat4 (NF4) Quantization + Double Quantization.
3. prepare_model_for_kbit_training() for stable low-rank gradient checkpointing.
4. Completion-only loss masking (-100 labels) for clinical responses.
5. PEFT LoRA Adapters (r=16, alpha=32) targeting all linear projection modules.
6. Weights & Biases (W&B) experiment logging.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from prepare_dataset import prepare_data

# Production 7B/8B Model for Single-GPU QLoRA
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # or meta-llama/Meta-Llama-3-8B-Instruct
OUTPUT_DIR = "./final_qlora_7b_adapter"
WANDB_PROJECT = "clinical-qlora-7b-production"


def get_bnb_4bit_config():
    """Information-theoretically optimal 4-bit NF4 Quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


def format_completion_only_tokens(samples, tokenizer, max_length=512):
    """Masks prompt tokens with -100 so loss is computed strictly on clinical responses."""
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for item in samples:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]
        context_str = f"\nContext: {input_text}" if input_text else ""

        prompt_str = (
            "<|im_start|>system\n"
            "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}{context_str}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        full_str = prompt_str + f"{output}<|im_end|>"

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = tokenizer.encode(full_str, add_special_tokens=False)

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        input_ids_list.append(full_ids)
        attention_mask_list.append([1] * len(full_ids))
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    })


def run_qlora_training():
    print("=" * 70)
    print("      🚀 PRODUCTION 4-BIT NF4 QLoRA TRAINING ON 7B FOUNDATION MODEL")
    print("=" * 70)

    # 1. Hardware Verification
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Hardware Accelerator: {device.upper()}")

    # 2. Weights & Biases Logging
    report_to = "none"
    if WANDB_AVAILABLE:
        try:
            if "WANDB_API_KEY" not in os.environ:
                os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=WANDB_PROJECT,
                name="qlora-qwen7b-clinical-run",
                config={"model": MODEL_NAME, "quant": "4-bit-nf4", "lora_r": 16, "lora_alpha": 32, "lr": 2e-4}
            )
            report_to = "wandb"
            print(f"📈 W&B Tracking Active (Mode: {os.environ.get('WANDB_MODE', 'online')})")
        except Exception as e:
            print(f"⚠️ W&B skipped: {e}")

    # 3. Load Tokenizer
    print(f"\n1. Loading Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load 4-bit Quantized Base Model
    print(f"2. Loading Quantized Model: {MODEL_NAME}...")
    if is_cuda:
        bnb_config = get_bnb_4bit_config()
        print("⚡ Applying BitsAndBytes 4-bit NF4 Quantization + Double Quant...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        print("🍎 [Local Development Mode]: BitsAndBytes CUDA kernels require NVIDIA GPU. Loading fallback on Mac MPS.")
        fallback_name = "Qwen/Qwen2.5-0.5B-Instruct"
        base_model = AutoModelForCausalLM.from_pretrained(
            fallback_name,
            torch_dtype=torch.float32
        ).to(device)

    # 5. Inject LoRA Adapters
    print("\n3. Injecting LoRA Adapters (r=16, alpha=32)...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    qlora_model = get_peft_model(base_model, peft_config)
    qlora_model.print_trainable_parameters()

    # 6. Load Full Dataset with Completion-Only Masking
    print("\n4. Loading & Formatting Full Domain Dataset...")
    data_splits = prepare_data("medical_domain_dataset.jsonl", test_ratio=0.20, seed=42)
    train_dataset = format_completion_only_tokens(data_splits["train"], tokenizer)
    eval_dataset = format_completion_only_tokens(data_splits["test"], tokenizer)
    print(f"  • Train Set: {len(train_dataset)} cases, Test Set: {len(eval_dataset)} cases")

    # 7. Collator & Arguments
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=qlora_model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir="./qlora_checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="no",
        report_to=report_to,
        remove_unused_columns=False
    )

    # 8. Train
    print("\n5. 🚀 Starting 4-Bit QLoRA SFT Training Loop...")
    trainer = Trainer(
        model=qlora_model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args
    )

    train_result = trainer.train()

    # 9. Save QLoRA Adapter Checkpoint
    print(f"\n6. 💾 Saving trained QLoRA adapter to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if report_to == "wandb" and WANDB_AVAILABLE:
        wandb.finish()

    print("=" * 70)
    print(f"🎉 QLoRA Training Complete! Final Train Loss: {train_result.metrics.get('train_loss', 0.0):.4f}")
    print(f"✅ Production Adapter Saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_qlora_training()
