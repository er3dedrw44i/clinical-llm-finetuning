"""
Step 2: Data Engineering & Preparation Module.
Pipeline:
1. Load raw JSONL (with JSON crash protection).
2. Validate required fields (instruction & output).
3. Deduplicate by (instruction, input) tuple.
4. Shuffle (seed=42).
5. Split into Train (80%) and Test (20%).
6. Output telemetry audit report.
"""

import json
import os
import random


def load_and_clean_data(file_path):
    """
    Loads JSONL, catches malformed JSON, validates non-empty fields,
    and deduplicates by (instruction, input). Returns clean data + stats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    total_loaded = 0
    invalid_count = 0
    duplicate_count = 0
    clean_records = []
    seen_keys = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_loaded += 1

            # 1. Crash Protection: Catch malformed JSON
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue

            # 2. Validation: Must have non-empty instruction and output
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            input_context = item.get("input", "").strip()

            if not instruction or not output:
                invalid_count += 1
                continue

            # 3. Context-Aware Deduplication: (instruction, input)
            dedup_key = (instruction, input_context)
            if dedup_key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(dedup_key)

            clean_records.append({
                "instruction": instruction,
                "input": input_context,
                "output": output
            })

    stats = {
        "loaded": total_loaded,
        "invalid": invalid_count,
        "duplicates": duplicate_count,
        "final": len(clean_records)
    }

    return clean_records, stats


def prepare_data(file_path="medical_domain_dataset.jsonl", test_ratio=0.20, seed=42):
    """
    Main Step 2 function: Cleans data and splits into Train (80%) and Test (20%).
    """
    clean_records, stats = load_and_clean_data(file_path)

    # Deterministic shuffle
    random.seed(seed)
    random.shuffle(clean_records)

    # 80% Train / 20% Test Split
    split_idx = int(len(clean_records) * (1.0 - test_ratio))
    if split_idx == len(clean_records) and len(clean_records) > 1:
        split_idx = len(clean_records) - 1

    train_set = clean_records[:split_idx]
    test_set = clean_records[split_idx:]

    # Telemetry Summary Report
    print("=" * 45)
    print("      📊 DATASET PREPARATION AUDIT REPORT")
    print("=" * 45)
    print(f"  • Total Loaded:      {stats['loaded']}")
    print(f"  • Invalid / Broken:  {stats['invalid']}")
    print(f"  • Duplicates Dropped: {stats['duplicates']}")
    print(f"  • Clean Records:     {stats['final']}")
    print("-" * 45)
    print(f"  • Train Set (80%):   {len(train_set)}")
    print(f"  • Test Set  (20%):   {len(test_set)}")
    print("=" * 45)

    return {
        "train": train_set,
        "test": test_set
    }


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(current_dir, "medical_domain_dataset.jsonl")

    splits = prepare_data(dataset_file, test_ratio=0.20)
    print("\n--- Sample Clean Record ---")
    print("Instruction:", splits["train"][0]["instruction"][:80] + "...")
    print("Output:", splits["train"][0]["output"][:80] + "...")
