"""
Shared Clinical Data & Tokenization Utilities.
Single Source of Truth for:
1. Parsing diagnostic options (extract_predicted_option)
2. Safe token formatting with prompt-protection & absolute max_length guarantee
3. Dataset loading and splitting
"""

import os
import re
import json
import math
from typing import List, Dict, Tuple, Any
from datasets import Dataset

DEFAULT_MAX_LENGTH = 512


def load_split(file_path: str) -> List[Dict[str, Any]]:
    """Loads records from a JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def extract_predicted_option(text: str) -> str:
    """
    Strictly extracts option letter (A, B, C, D, E) from model generation.
    Supports both direct options (A:, B.) and conversational formulations (Choice B, most likely diagnosis is C).
    Guarantees that phrases like 'A 68-year-old male...' are NOT falsely parsed as option A.
    """
    if not text:
        return "NONE"
    clean = text.strip()

    # Pattern 1: Leading option with delimiter (A:, A., A), A -, (A))
    m1 = re.match(r'^\s*\(?([A-Ea-e])\)?\s*[:\.\)\-]\s*', clean)
    if m1:
        return m1.group(1).upper()

    # Pattern 2: Explicit answer phrasing (Option A, Choice B, Answer: C, The correct answer is D, most likely diagnosis is E)
    m2 = re.search(r'(?:option|choice|answer|diagnosis|pathogen|most likely(?: diagnosis| pathogen)?)\s*(?:is\s*)?[:\s\-]*\(?([A-Ea-e])\)?(?:\b|[\.\:\)\-])', clean, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    # Pattern 3: Concluding statement (Therefore, (B) is..., Thus, answer C)
    m3 = re.search(r'(?:therefore|thus|hence)\s*,?\s*(?:the\s*)?(?:correct\s*)?(?:option|choice|answer|diagnosis)?\s*(?:is\s*)?\(?([A-Ea-e])\)?(?:\b|[\.\:\)\-])', clean, re.IGNORECASE)
    if m3:
        return m3.group(1).upper()

    # Pattern 4: Standalone single-token output
    tokens = clean.split()
    if len(tokens) == 1:
        tok = tokens[0].upper().rstrip('.:,)')
        if tok in ["A", "B", "C", "D", "E"]:
            return tok

    return "NONE"


def format_single_example_tokens(
    prompt_str: str,
    output_str: str,
    tokenizer,
    max_length: int = DEFAULT_MAX_LENGTH
) -> Tuple[List[int], List[int], int]:
    """
    Formats a single prompt-completion pair with strict completion loss masking (-100).
    Guarantees:
    1. len(full_ids) <= max_length (absolute bound)
    2. Completion tokens are protected; prompt is truncated from left if needed.
    3. If output itself exceeds max_length, output is safely truncated to max_length - 1.
    """
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    output_ids = tokenizer.encode(output_str + "<|im_end|>", add_special_tokens=False)

    # Edge Case: Output itself exceeds max_length
    if len(output_ids) >= max_length:
        output_ids = output_ids[:max_length - 1]

    # If combined length exceeds max_length, preserve output and truncate prompt from left
    if len(prompt_ids) + len(output_ids) > max_length:
        max_prompt_len = max(1, max_length - len(output_ids))
        prompt_ids = prompt_ids[-max_prompt_len:]

    full_ids = prompt_ids + output_ids
    labels = [-100] * len(prompt_ids) + output_ids

    assert len(full_ids) <= max_length, f"Length {len(full_ids)} exceeds max_length {max_length}"
    assert len(full_ids) == len(labels), "full_ids and labels length mismatch"

    return full_ids, labels, len(prompt_ids)


def format_completion_only_dataset(
    samples: List[Dict[str, Any]],
    tokenizer,
    max_length: int = DEFAULT_MAX_LENGTH
) -> Dataset:
    """Formats an entire list of samples into a Hugging Face Dataset with -100 masking."""
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for item in samples:
        inst = item["instruction"]
        inp = item.get("input", "")
        out = item["output"]
        context_str = f"\nContext: {inp}" if inp else ""

        prompt_str = (
            "<|im_start|>system\n"
            "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
            f"<|im_start|>user\n{inst}{context_str}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        full_ids, labels, _ = format_single_example_tokens(prompt_str, out, tokenizer, max_length=max_length)

        input_ids_list.append(full_ids)
        attention_mask_list.append([1] * len(full_ids))
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    })
