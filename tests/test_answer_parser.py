import re
import unittest

def extract_predicted_option(text: str) -> str:
    """
    Strictly extracts option letter (A, B, C, D, E) from model generation.
    Guarantees that phrases like 'A 68-year-old male...' or 'A patient presents...'
    are NOT falsely parsed as option A.
    """
    if not text:
        return "NONE"
    
    clean = text.strip()
    
    # 1. Option with explicit delimiter: 'A:', 'A.', 'A)', 'A -', '(A)'
    m1 = re.match(r'^\s*\(?([A-Ea-e])\)?\s*[:\.\)\-]\s*', clean)
    if m1:
        return m1.group(1).upper()

    # 2. Explicit answer phrasing: 'Option A', 'Answer: A', 'The correct answer is (B)'
    m2 = re.search(r'(?:option|answer(?:\s*is)?)\s*[:\s\-]*\(?([A-Ea-e])\)?(?:\b|[\.\:\)\-])', clean, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    # 3. Exact standalone single-token output: 'A', 'B', 'C', 'D', 'E'
    tokens = clean.split()
    if len(tokens) == 1:
        tok = tokens[0].upper().rstrip('.:,)')
        if tok in ["A", "B", "C", "D", "E"]:
            return tok

    return "NONE"


class TestAnswerParser(unittest.TestCase):
    def test_valid_option_formats(self):
        self.assertEqual(extract_predicted_option("B: Streptococcus pneumoniae pneumonia"), "B")
        self.assertEqual(extract_predicted_option("C. Formation of C5-9 complex"), "C")
        self.assertEqual(extract_predicted_option("(D) Nitrofurantoin"), "D")
        self.assertEqual(extract_predicted_option("A - High fever"), "A")
        self.assertEqual(extract_predicted_option("B"), "B")
        self.assertEqual(extract_predicted_option("The correct answer is Option B: Empagliflozin"), "B")
        self.assertEqual(extract_predicted_option("Answer: C"), "C")

    def test_rejection_of_false_positives(self):
        # 'A' as an article in clinical vignettes must return NONE
        self.assertEqual(extract_predicted_option("A 68-year-old male with fever"), "NONE")
        self.assertEqual(extract_predicted_option("A patient presents with acute abdominal pain"), "NONE")
        self.assertEqual(extract_predicted_option("An elderly female with history of diabetes"), "NONE")
        
        # 'b' inside words must return NONE
        self.assertEqual(extract_predicted_option("Staphylococcus aureus pneumonia"), "NONE")
        self.assertEqual(extract_predicted_option("Patient should be treated with cephalosporin"), "NONE")
        self.assertEqual(extract_predicted_option("No obvious abnormalities noted"), "NONE")


if __name__ == "__main__":
    unittest.main()
