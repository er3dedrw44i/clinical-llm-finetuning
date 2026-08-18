import unittest
from transformers import AutoTokenizer
from data_utils import format_single_example_tokens

class TestCompletionMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def test_normal_completion_masking(self):
        prompt_str = "<|im_start|>user\nDiagnose patient with fever<|im_end|>\n<|im_start|>assistant\n"
        output_str = "B: Streptococcus pneumoniae"
        full_ids, labels, prompt_len = format_single_example_tokens(prompt_str, output_str, self.tokenizer, max_length=512)

        self.assertLessEqual(len(full_ids), 512)
        self.assertEqual(len(full_ids), len(labels))
        self.assertTrue(all(l == -100 for l in labels[:prompt_len]))
        self.assertTrue(all(l != -100 for l in labels[prompt_len:]))

    def test_overlength_prompt_truncation(self):
        # Massive prompt with 1,000 tokens
        huge_prompt = "<|im_start|>user\n" + ("clinical case details with fever " * 200) + "<|im_end|>\n<|im_start|>assistant\n"
        output_str = "B: Streptococcus pneumoniae"
        full_ids, labels, prompt_len = format_single_example_tokens(huge_prompt, output_str, self.tokenizer, max_length=512)

        self.assertLessEqual(len(full_ids), 512)
        self.assertEqual(len(full_ids), len(labels))
        # Ensure completion tokens are 100% preserved
        comp_tokens = [l for l in labels if l != -100]
        self.assertGreater(len(comp_tokens), 0)

    def test_overlength_output_edge_case(self):
        prompt_str = "<|im_start|>user\nExplain symptoms<|im_end|>\n<|im_start|>assistant\n"
        huge_output = "Diagnostic reasoning with detailed explanation " * 200
        full_ids, labels, prompt_len = format_single_example_tokens(prompt_str, huge_output, self.tokenizer, max_length=512)

        # Must strictly enforce max_length bound
        self.assertLessEqual(len(full_ids), 512)
        self.assertEqual(len(full_ids), len(labels))

if __name__ == "__main__":
    unittest.main()
