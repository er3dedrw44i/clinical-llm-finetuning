import unittest
from transformers import AutoTokenizer

class TestCompletionMask(unittest.TestCase):
    def test_completion_loss_masking(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        instruction = "Diagnose the patient."
        input_text = "Patient has fever and cough."
        output = "B: Streptococcus pneumoniae"

        prompt_str = (
            "<|im_start|>system\n"
            "You are an expert Clinical Medicine AI assistant. Provide accurate, evidence-based guidance.<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\nContext: {input_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        full_str = prompt_str + f"{output}<|im_end|>"

        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = tokenizer.encode(full_str, add_special_tokens=False)

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        self.assertEqual(len(labels), len(full_ids), "Labels and full_ids length mismatch")
        self.assertTrue(all(l == -100 for l in labels[:prompt_len]), "Prompt tokens must be masked with -100")
        self.assertTrue(any(l != -100 for l in labels[prompt_len:]), "Completion tokens must have non-negative labels")

if __name__ == "__main__":
    unittest.main()
