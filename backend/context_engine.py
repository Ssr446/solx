from typing import Dict, Any

class ContextResult:
    def __init__(self, phrase: str, source: str, explanation: str, severity: str):
        self.phrase = phrase
        self.source = source
        self.explanation = explanation
        self.severity = severity
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase,
            "source": self.source,
            "explanation": self.explanation,
            "severity": self.severity
        }

class CulturalContextEngine:
    def __init__(self):
        # Knowledge base of Tamil pop-culture references, memes, and dialogues
        self.kb_dialogues = {
            "appatakar": {
                "source": "Tanglish meme template (2019 onwards)",
                "explanation": "Sarcastic title for someone pretending to be superior or a know-it-all.",
                "severity": "Low (Sarcasm)"
            },
            "enna koduma saravanan idhu": {
                "source": "Prabhu's dialogue from Chandramukhi (2005)",
                "explanation": "Used to express mock despair or frustration at a ridiculous situation.",
                "severity": "Low (Sarcasm)"
            },
            "sothula uppu potu thana sapudra": {
                "source": "Common Tamil proverb/dialogue",
                "explanation": "Questioning someone's basic common sense or humanity.",
                "severity": "Medium (Insult/Sarcasm)"
            },
            "goyyale": {
                "source": "Chennai local slang",
                "explanation": "A mild swear word used to express anger or frustration.",
                "severity": "Medium (Toxic)"
            },
            "poramboku": {
                "source": "Tamil slang",
                "explanation": "Derogatory term referring to someone who is useless or a vagabond. Originally referred to unassigned lands.",
                "severity": "High (Cyberbullying/Toxic)"
            },
            "dubakoor": {
                "source": "Tamil slang",
                "explanation": "Refers to a fraud, fake person, or something of very poor quality.",
                "severity": "Medium (Toxic)"
            },
             "loose ah da nee": {
                "source": "Common Tamil phrase",
                "explanation": "Direct insult questioning mental stability ('Are you mad?').",
                "severity": "Medium (Toxic)"
            }
        }
        
    def lookup(self, phrase: str) -> ContextResult:
        """
        Looks up a phrase in the knowledge base and returns its cultural context.
        """
        phrase_lower = phrase.lower()
        
        # Check exact matches or substrings
        for key, data in self.kb_dialogues.items():
            if key in phrase_lower:
                return ContextResult(
                    phrase=key,
                    source=data["source"],
                    explanation=data["explanation"],
                    severity=data["severity"]
                )
                
        # Fallback if no specific cultural context is found
        return ContextResult(
            phrase=phrase,
            source="General Language",
            explanation="No specific pop-culture reference detected.",
            severity="Unknown"
        )
        
    def explain(self, ref: str) -> str:
        """Helper to just get the explanation string"""
        result = self.lookup(ref)
        return result.explanation
        
    def get_severity(self) -> str:
        """Placeholder for severity context"""
        return "Unknown"

# Test the engine
if __name__ == "__main__":
    engine = CulturalContextEngine()
    res = engine.lookup("avaru periya appatakar da")
    print(res.to_dict())
