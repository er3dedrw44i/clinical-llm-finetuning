"""
Production 4-Bit NF4 QLoRA Training Engine (NVIDIA CUDA / Cloud).
Configured for reproducible single-GPU training on 4,000 USMLE training cases.
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
from prepare_dataset import load_split

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./final_qlora_7b_adapter"
TRAIN_DATA_PATH = "train.jsonl"


def get_bnb_4bit_config():
    """Hardware-aware 4-bit NormalFloat4 (NF4) Quantization."""
    # Native FP16 on Tesla T4 (Turing CC 7.5); BF16 on Ampere/Hopper (CC 8.0+)
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


def format_completion_only_tokens(samples, tokenizer, max_length=512):
    """
    Masks prompt tokens with -100 while guaranteeing 100% completion token preservation.
    If full sequence exceeds max_length, truncates prompt from left to protect answer loss.
    """
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

        # Truncate prompt from left if needed to protect 100% of the output completion
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


def run_qlora_training():
    print("=" * 70)
    print("      🚀 PRODUCTION 4-BIT NF4 QLoRA TRAINING ON 7B FOUNDATION MODEL")
    print("=" * 70)

    # 1. Hardware Verification
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Hardware Accelerator: {device.upper()}")

    # 2. Load Tokenizer
    print(f"\n1. Loading Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load 4-bit Quantized Base Model
    print(f"2. Loading Quantized Model: {MODEL_NAME}...")
    if is_cuda:
        bnb_config = get_bnb_4bit_config()
        print(f"⚡ Applying BitsAndBytes 4-bit NF4 Quantization (Compute Dtype: {bnb_config.bnb_4bit_compute_dtype})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True
        )
    else:
        print("🍎 [Local Development Mode]: BitsAndBytes CUDA kernels require NVIDIA GPU. Loading fallback on Mac MPS.")
        fallback_name = "Qwen/Qwen2.5-0.5B-Instruct"
        base_model = AutoModelForCausalLM.from_pretrained(
            fallback_name,
            torch_dtype=torch.float32
        ).to(device)

    # 4. Inject LoRA Adapters
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

    # 5. Load Persistent train.jsonl Dataset ONLY
    print(f"\n4. Loading Training Data strictly from {TRAIN_DATA_PATH}...")
    train_data = load_split(TRAIN_DATA_PATH)
    train_dataset = format_completion_only_tokens(train_data, tokenizer, max_length=512)
    print(f"  • Training Dataset Size: {len(train_dataset)} cases (100% completion coverage guaranteed)")

    # 6. Dynamic Padding Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=qlora_model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    # 7. Training Arguments with PagedAdamW, Checkpoints & Gradient Checkpointing
    training_args = TrainingArguments(
        output_dir="./qlora_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=25,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        fp16=is_cuda,
        optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False
    )

    # 8. Reset Peak Memory Stats & Train
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    print("\n5. 🚀 Starting 4-Bit QLoRA SFT Training Loop (250 total optimizer steps)...")
    trainer = Trainer(
        model=qlora_model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args
    )

    train_result = trainer.train()

    # 9. Measure Clean Peak VRAM & Save Checkpoint
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else 0.0
    print(f"\n6. 💾 Saving trained QLoRA adapter to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 70)
    print(f"🎉 QLoRA Training Complete! Final Train Loss: {train_result.metrics.get('train_loss', 0.0):.4f}")
    print(f"📊 Observed Peak Training VRAM: {peak_vram:.2f} GB on {device.upper()} (Measured via reset_peak_memory_stats())")
    print(f"✅ Production Adapter Saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_qlora_training()
