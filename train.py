"""
Step 5: Experiment A Baseline SFT Training on Apple Silicon M5 GPU.
Features:
1. Completion-Only Loss Masking (-100 on prompt tokens).
2. Safe token formatting preserving 100% of diagnostic completion tokens.
3. Loads strictly from persistent `train.jsonl` (4,000 cases).
4. Peft LoRA Adapters (r=16, alpha=32).
"""

import os
import time
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
from prepare_dataset import load_split

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./final_adapter"
TRAIN_DATA_PATH = "train.jsonl"


def format_completion_only_tokens(samples, tokenizer, max_length=512):
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
        output_str = f"{output}<|im_end|>"

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)

        # Protect 100% of completion loss signal by truncating prompt from left if needed
        if len(prompt_ids) + len(output_ids) > max_length:
            max_prompt_len = max(10, max_length - len(output_ids))
            prompt_ids = prompt_ids[-max_prompt_len:]

        full_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids

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
    print("      🚀 STEP 5: EXPERIMENT A SFT TRAINING ON APPLE SILICON GPU (MPS)")
    print("=" * 70)

    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🍎 Active Hardware Accelerator: {device.upper()}")

    # 1. Load Tokenizer & Model
    print(f"\n1. Loading Base Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device)

    # 2. Inject LoRA Adapters
    print("\n2. Injecting LoRA Adapters (r=16, alpha=32)...")
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

    # 3. Load Persistent Training Split ONLY
    print(f"\n3. Formatting Training Cases from {TRAIN_DATA_PATH}...")
    train_data = load_split(TRAIN_DATA_PATH)
    train_dataset = format_completion_only_tokens(train_data, tokenizer, max_length=512)
    print(f"  • Active Training Cases: {len(train_dataset)} (100% completion preservation guaranteed)")

    # 4. Collator & Training Arguments
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=lora_model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir="./lora_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=25,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=lora_model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args
    )

    print("\n4. 🚀 Starting Experiment A Baseline Training Loop...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    # 5. Save Adapter
    os.makedirs(ADAPTER_PATH, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)

    print("=" * 70)
    print(f"🎉 Training Complete in {elapsed:.2f}s! Final Loss: {train_result.metrics.get('train_loss', 0.0):.4f}")
    print(f"💾 Trained LoRA Adapter Saved to: {ADAPTER_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_training()