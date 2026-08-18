import re
import unittest

def extract_predicted_option(text: str) -> str:
    """
    Strictly extracts option letter (A, B, C, D, E) from model generation.
    Rejects accidental substring occurrences (e.g., 'b' in 'Staphylococcus').
    """
    if not text:
        return "NONE"
    
    clean_text = text.strip()
    
    # Pattern 1: Leading option letter (e.g. 'A: ...', 'B. ...', '(C)', 'D - ...')
    m1 = re.match(r'^\s*\(?([A-Ea-e])\)?(?:\s*[:\.\)\-]|\s+|$)', clean_text)
    if m1:
        return m1.group(1).upper()

    # Pattern 2: Explicit answer phrasing (e.g. 'The correct answer is B', 'Option C')
    m2 = re.search(r'(?:option|answer(?:\s*is)?)\s*[:\s\-]*\(?([A-Ea-e])\)?(?:\b|[\.\:\)])', clean_text, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    # Pattern 3: Standalone first token is a valid option letter
    first_tok = clean_text.split()[0].upper().rstrip('.:,)') if clean_text.split() else ""
    if first_tok in ["A", "B", "C", "D", "E"]:
        return first_tok

    return "NONE"


class TestAnswerParser(unittest.TestCase):
    def test_clean_option_prefix(self):
        self.assertEqual(extract_predicted_option("B: Streptococcus pneumoniae pneumonia"), "B")
        self.assertEqual(extract_predicted_option("C. Formation of C5-9 complex"), "C")
        self.assertEqual(extract_predicted_option("(D) Nitrofurantoin"), "D")
        self.assertEqual(extract_predicted_option("A - High fever"), "A")

    def test_conversational_phrasing(self):
        self.assertEqual(extract_predicted_option("The correct answer is Option B: Empagliflozin"), "B")
        self.assertEqual(extract_predicted_option("Answer: C"), "C")

    def test_rejection_of_accidental_substrings(self):
        # 'b' inside Staphylococcus must NOT match option B
        self.assertEqual(extract_predicted_option("Staphylococcus aureus pneumonia"), "NONE")
        self.assertEqual(extract_predicted_option("Patient should be treated with cephalosporin"), "NONE")
        self.assertEqual(extract_predicted_option("No obvious abnormalities noted"), "NONE")


if __name__ == "__main__":
    unittest.main()
