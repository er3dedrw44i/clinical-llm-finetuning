import os
import json
import hashlib
import unittest

TRAIN_PATH = "train.jsonl"
TEST_PATH = "test.jsonl"

def get_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

class TestDatasetSplit(unittest.TestCase):
    def test_split_files_exist(self):
        self.assertTrue(os.path.exists(TRAIN_PATH), "train.jsonl does not exist")
        self.assertTrue(os.path.exists(TEST_PATH), "test.jsonl does not exist")

    def test_split_counts(self):
        with open(TRAIN_PATH, "r", encoding="utf-8") as f:
            train_lines = [l for l in f if l.strip()]
        with open(TEST_PATH, "r", encoding="utf-8") as f:
            test_lines = [l for l in f if l.strip()]

        self.assertEqual(len(train_lines), 4000, f"Expected 4000 train lines, got {len(train_lines)}")
        self.assertEqual(len(test_lines), 1000, f"Expected 1000 test lines, got {len(test_lines)}")

    def test_zero_leakage_hash_overlap(self):
        train_hashes = set()
        with open(TRAIN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = f"{rec.get('instruction','')}|{rec.get('input','')}|{rec.get('output','')}"
                train_hashes.add(get_hash(key))

        test_hashes = set()
        with open(TEST_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = f"{rec.get('instruction','')}|{rec.get('input','')}|{rec.get('output','')}"
                test_hashes.add(get_hash(key))

        overlap = train_hashes.intersection(test_hashes)
        self.assertEqual(len(overlap), 0, f"Detected data leakage! {len(overlap)} overlapping records found between train and test.")

if __name__ == "__main__":
    unittest.main()
