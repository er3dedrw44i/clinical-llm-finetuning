"""
Step 2: Production Dataset Preparation, Validation, and Deterministic Splitting.
Materializes persistent `train.jsonl` (80%) and `test.jsonl` (20%) on disk
to guarantee 100% data separation with zero data leakage.
"""

import os
import json
import random
import hashlib

DATASET_PATH = "medical_domain_dataset.jsonl"
TRAIN_PATH = "train.jsonl"
TEST_PATH = "test.jsonl"
SPLIT_SEED = 42


def get_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def prepare_and_materialize_splits(
    input_file: str = DATASET_PATH,
    train_file: str = TRAIN_PATH,
    test_file: str = TEST_PATH,
    test_ratio: float = 0.20,
    seed: int = SPLIT_SEED
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Source dataset not found at {input_file}")

    seen_hashes = set()
    clean_records = []
    invalid_count = 0
    duplicate_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                instruction = record.get("instruction", "").strip()
                input_text = record.get("input", "").strip()
                output = record.get("output", "").strip()

                if not instruction or not output:
                    invalid_count += 1
                    continue

                content_hash = get_hash(f"{instruction}|{input_text}|{output}")
                if content_hash in seen_hashes:
                    duplicate_count += 1
                    continue

                seen_hashes.add(content_hash)
                clean_records.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output
                })
            except json.JSONDecodeError:
                invalid_count += 1

    # Deterministic Shuffle and Split
    random.seed(seed)
    random.shuffle(clean_records)

    total_clean = len(clean_records)
    test_size = int(total_clean * test_ratio)
    train_size = total_clean - test_size

    train_data = clean_records[:train_size]
    test_data = clean_records[train_size:]

    # Write persistent physical files
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("      📊 PERSISTENT DATASET SPLIT AUDIT REPORT")
    print("=" * 60)
    print(f"  • Source File:          {input_file}")
    print(f"  • Total Loaded:         {line_num}")
    print(f"  • Invalid / Dropped:    {invalid_count}")
    print(f"  • Duplicates Dropped:   {duplicate_count}")
    print(f"  • Clean Records:        {total_clean}")
    print("-" * 60)
    print(f"  • Saved Train File:     {train_file} ({len(train_data)} records - {100-int(test_ratio*100)}%)")
    print(f"  • Saved Test File:      {test_file} ({len(test_data)} records - {int(test_ratio*100)}%)")
    print(f"  • Deterministic Seed:   {seed}")
    print("=" * 60)

    return {"train": train_data, "test": test_data}


def load_split(file_path: str):
    """Loads a materialized split from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file {file_path} not found. Run prepare_dataset.py first.")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


if __name__ == "__main__":
    prepare_and_materialize_splits()
