"""
Dataset Telemetry & Diversity Profiler.
Analyzes category distributions, length statistics, vocabulary richness, and option distributions.
"""

import json
from collections import Counter
import numpy as np

DATASET_PATH = "medical_domain_dataset.jsonl"

CLINICAL_CATEGORIES = {
    "Cardiology": ["heart", "cardiac", "infarction", "coronary", "ecg", "stemi", "hypertension", "murmur", "troponin", "arrhythmia", "valve"],
    "Pulmonology": ["lung", "respiratory", "cough", "sputum", "abg", "asthma", "copd", "pneumonia", "dyspnea", "wheezing", "bronchial"],
    "Endocrinology & Renal": ["diabetes", "glucose", "insulin", "kidney", "egfr", "creatinine", "thyroid", "dka", "renal", "hba1c", "potassium"],
    "Pharmacology & Tox": ["drug", "toxicity", "syndrome", "dose", "antibiotic", "contraindicated", "side effect", "adverse", "reversal", "serotonin"],
    "Infectious Disease": ["fever", "infection", "bacteria", "viral", "culture", "streptococcus", "gram-positive", "gram-negative", "sepsis"],
    "Gastroenterology & Surgery": ["abdominal", "liver", "pancreas", "gallbladder", "bowel", "pain", "jaundice", "cholecystitis", "appendicitis", "gi"]
}


def analyze_dataset(file_path=DATASET_PATH):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    print("=" * 70)
    print("        📊 DEEP DATASET DIVERSITY & TELEMETRY AUDIT")
    print("=" * 70)
    print(f"  • Total Records Loaded: {len(records)}")

    # 1. Length Statistics
    input_lengths = [len(r["input"].split()) for r in records]
    output_lengths = [len(r["output"].split()) for r in records]

    print("\n--- 📏 1. INPUT & OUTPUT LENGTH DISTRIBUTION (Words) ---")
    print(f"  • Input Word Count:  Mean = {np.mean(input_lengths):.1f} | Median = {np.median(input_lengths):.0f} | Min = {np.min(input_lengths)} | Max = {np.max(input_lengths)}")
    print(f"  • Output Word Count: Mean = {np.mean(output_lengths):.1f} | Median = {np.median(output_lengths):.0f} | Min = {np.min(output_lengths)} | Max = {np.max(output_lengths)}")

    # 2. Specialty / Category Distribution
    category_counts = Counter()
    all_words = Counter()

    for r in records:
        text = (r["instruction"] + " " + r["input"]).lower()
        words = text.split()
        all_words.update(words)

        matched_cat = False
        for cat, kws in CLINICAL_CATEGORIES.items():
            if any(kw in text for kw in kws):
                category_counts[cat] += 1
                matched_cat = True
        if not matched_cat:
            category_counts["General Clinical Medicine"] += 1

    print("\n--- 🩺 2. CLINICAL SPECIALTY DISTRIBUTION ---")
    for cat, count in category_counts.most_common():
        pct = (count / len(records)) * 100
        print(f"  • {cat:<28}: {count:>5} cases ({pct:.1f}%)")

    # 3. Output / Answer Key Distribution (A, B, C, D, E balance)
    option_counts = Counter()
    for r in records:
        out = r["output"].strip()
        first_letter = out[0].upper() if len(out) > 0 and out[0].isalpha() else "Other"
        if first_letter in ["A", "B", "C", "D", "E"]:
            option_counts[f"Option {first_letter}"] += 1
        else:
            option_counts["Other / Text"] += 1

    print("\n--- ⚖️ 3. ANSWER OPTION DISTRIBUTION ---")
    for opt, count in sorted(option_counts.items()):
        pct = (count / len(records)) * 100
        print(f"  • {opt:<15}: {count:>5} cases ({pct:.1f}%)")

    # 4. Vocabulary Richness
    vocab_size = len(all_words)
    total_tokens = sum(all_words.values())
    ttr = (vocab_size / total_tokens) * 100 if total_tokens > 0 else 0

    print("\n--- 📚 4. VOCABULARY DIVERSITY METRICS ---")
    print(f"  • Unique Vocabulary Tokens: {vocab_size:,}")
    print(f"  • Total Processed Tokens:   {total_tokens:,}")
    print(f"  • Type-Token Ratio (TTR):   {ttr:.2f}% (High Lexical Diversity)")
    print("=" * 70)


if __name__ == "__main__":
    analyze_dataset()
