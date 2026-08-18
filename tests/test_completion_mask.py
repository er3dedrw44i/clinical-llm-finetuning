import unittest

class TestCompletionMask(unittest.TestCase):
    def test_shared_token_formatting_logic(self):
        try:
            from transformers import AutoTokenizer
            from data_utils import format_single_example_tokens
            
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            prompt_str = "<|im_start|>user\nDiagnose patient with fever<|im_end|>\n<|im_start|>assistant\n"
            output_str = "B: Streptococcus pneumoniae"
            full_ids, labels, prompt_len = format_single_example_tokens(prompt_str, output_str, tokenizer, max_length=512)

            self.assertLessEqual(len(full_ids), 512)
            self.assertEqual(len(full_ids), len(labels))
            self.assertTrue(all(l == -100 for l in labels[:prompt_len]))
            self.assertTrue(all(l != -100 for l in labels[prompt_len:]))
        except ImportError:
            # When transformers is not installed in local standard library environment
            self.skipTest("transformers not installed in local environment; skipping live tokenizer test.")

if __name__ == "__main__":
    unittest.main()
