import os
import pickle
from typing import Dict


class SOLXClassifier:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.labels = ["Toxic", "Sarcastic", "Cyberbullying"]

        if not self.use_mock:
            model_path = os.path.join(os.path.dirname(__file__), "solx_model.pkl")
            if os.path.exists(model_path):
                print("Loading custom Tanglish ML model...")
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                print("Model loaded successfully.")
            else:
                print("Model file not found. Falling back to mock.")
                self.use_mock = True

    def predict(self, normalized_text: str) -> Dict[str, float]:
        """
        Takes normalized Tanglish text and returns confidence scores (0-100%).
        Uses the trained scikit-learn pipeline — no hardcoded rules.
        """
        if self.use_mock:
            return self._mock_predict(normalized_text)

        try:
            text_lower = normalized_text.lower()
            # predict_proba returns a list of arrays, one per output label
            # Each array shape: (n_samples, n_classes) where index 1 = P(positive)
            probabilities = self.model.predict_proba([text_lower])

            scores = {}
            for i, label in enumerate(self.labels):
                prob_true = probabilities[i][0][1]
                scores[label] = round(prob_true * 100, 1)

            return scores
        except Exception as e:
            print(f"Inference error: {e}")
            return {"Toxic": 0.0, "Sarcastic": 0.0, "Cyberbullying": 0.0, "error_msg": str(e)}

    def _mock_predict(self, text: str) -> Dict[str, float]:
        import random
        text = text.lower()
        scores = {
            "Toxic": random.uniform(5.0, 15.0),
            "Sarcastic": random.uniform(5.0, 15.0),
            "Cyberbullying": random.uniform(0.0, 10.0)
        }
        if "aptakar" in text or "mokka" in text:
            scores["Sarcastic"] = random.uniform(75.0, 95.0)
        if "poramboku" in text or "loosu" in text or "punda" in text or "bitch" in text:
            scores["Toxic"] = random.uniform(80.0, 98.0)
            scores["Cyberbullying"] = random.uniform(60.0, 85.0)
        for key in scores:
            scores[key] = min(round(scores[key], 1), 100.0)
        return scores


if __name__ == "__main__":
    clf = SOLXClassifier(use_mock=False)
    print(clf.predict("hey bitch how are you"))
    print(clf.predict("poda punda"))
    print(clf.predict("hello how are you"))
