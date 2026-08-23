import re

class TanglishNormalizer:
    def __init__(self):
        # Phonetic mapping rules to standard root words
        self.phonetic_map = {
            r"\b(y?e+n+a+)\b": "enna",
            r"\b(y?e+n+a+d+h+u+)\b": "ennadhu",
            r"\b(v+a+n+t+h+u+)\b": "vandhu",
            r"\b(p+a+n+n+u+)\b": "pannu",
            r"\b(p+o+c+h+u+)\b": "pochu",
            r"\b(i+r+u+k+k+u+)\b": "irukku",
            r"\b(s+e+i+y+a+)\b": "seiya",
            r"\b(k+e+t+u+k+k+o+)\b": "ketuko"
        }
        
        # Slang and abbreviation lookup table
        self.slang_table = {
            "appatakar": "aptakar",
            "mokka": "boring",
            "vetti": "useless",
            "loosu": "fool",
            "mental": "mad",
            "gm": "good morning",
            "gn": "good night",
            "bro": "brother",
            "macha": "friend",
            "machan": "friend",
            "da": "friend (informal)",
            "di": "friend (informal/female)",
            "gaandu": "irritated",
            "kaduppu": "annoyed"
        }

    def normalize(self, text: str) -> str:
        """
        Normalizes Tanglish text by fixing phonetic variations and slang.
        """
        if not text:
            return ""
            
        text = text.lower()
        
        # 1. Phonetic mapping using regex
        for pattern, replacement in self.phonetic_map.items():
            text = re.sub(pattern, replacement, text)
            
        # 2. Slang resolution
        words = text.split()
        normalized_words = [self.slang_table.get(word, word) for word in words]
        
        return " ".join(normalized_words)
        
    def map_variants(self, word: str) -> str:
        """Helper to test a single word"""
        return self.normalize(word)
        
    def detect_slang(self, token: str) -> str:
        """Returns the meaning of a slang word if it exists, otherwise the word itself"""
        return self.slang_table.get(token.lower(), token)

# Test the normalizer
if __name__ == "__main__":
    normalizer = TanglishNormalizer()
    sample = "avaru periya appatakar da, yenna pannuvanunga"
    print(f"Original: {sample}")
    print(f"Normalized: {normalizer.normalize(sample)}")
