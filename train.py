"""
Step 5: Production SFT Training with Strict Completion-Only Loss Masking (-100).
Ensures gradients and Cross-Entropy Loss are computed STRICTLY on assistant clinical responses,
preventing catastrophic forgetting of general instruction-following capabilities.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Optional Weights & Biases tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from prepare_dataset import prepare_data

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./final_adapter"
WANDB_PROJECT = "clinical-llm-finetuning"


def format_completion_only_tokens(samples, tokenizer, max_length=512):
    """
    Encodes (Prompt + Completion) and sets labels = -100 for all prompt tokens.
    PyTorch's CrossEntropyLoss automatically ignores label tokens with value -100.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

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

        # Truncate if needed
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        # Build labels with -100 for prompt tokens
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


def run_training():
    print("=" * 70)
    print("   🚀 STEP 5: SFT TRAINING WITH COMPLETION-ONLY LOSS MASKING (-100)")
    print("=" * 70)

    # 1. Detect Device
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🍎 Active Device: {device}")

    # 2. Initialize Weights & Biases (Offline Fallback)
    report_to = "none"
    if WANDB_AVAILABLE:
        try:
            if "WANDB_API_KEY" not in os.environ:
                os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=WANDB_PROJECT,
                name="qwen0.5b-completion-masking-run",
                config={"model": MODEL_NAME, "lora_r": 16, "lora_alpha": 32, "lr": 2e-4, "device": device}
            )
            report_to = "wandb"
            print(f"📈 Weights & Biases Connected (Mode: {os.environ.get('WANDB_MODE', 'online')})")
        except Exception as e:
            print(f"⚠️ W&B skipped: {e}")

    # 3. Load Tokenizer & Base Model
    print("\n1. Loading Tokenizer & Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device)

    # 4. Inject LoRA Adapters
    print("2. Injecting LoRA Adapters (r=16, alpha=32)...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()

    # 5. Prepare Dataset with Completion-Only Masking
    print("\n3. Formatting dataset with Completion-Only Loss Masking...")
    data_splits = prepare_data("medical_domain_dataset.jsonl", test_ratio=0.20, seed=42)
    train_dataset = format_completion_only_tokens(data_splits["train"], tokenizer)
    eval_dataset = format_completion_only_tokens(data_splits["test"], tokenizer)
    print(f"  • Train Samples: {len(train_dataset)}, Test Samples: {len(eval_dataset)}")

    # 6. Data Collator with Dynamic Padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=lora_model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="no",
        report_to=report_to,
        remove_unused_columns=False
    )

    # 8. Standard Trainer (Robust & Full PyTorch Control)
    print("\n4. Initializing Trainer & Executing SFT...")
    trainer = Trainer(
        model=lora_model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args
    )

    train_result = trainer.train()

    # 9. Save LoRA Adapter
    print(f"\n5. 💾 Saving trained LoRA adapter to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if report_to == "wandb" and WANDB_AVAILABLE:
        wandb.finish()

    print("=" * 70)
    print(f"🎉 Training Complete! Final Train Loss: {train_result.metrics.get('train_loss', 0.0):.4f}")
    print(f"✅ Adapter saved in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_training()