import joblib
import numpy as np
from transformers import pipeline

class ClassifierService:
    def __init__(self, language_model_path: str, emotion_model_path: str, device: int, confidence_threshold: float = 0.7):
        self.language_model = joblib.load(language_model_path)
        self.emotion_model = pipeline(
            "text-classification",
            model=emotion_model_path,
            tokenizer=emotion_model_path,
            device=device,
            truncation=True,
            max_length=512
        )
        self.confidence_threshold = confidence_threshold            

    def detect_language(self, text: str) -> str:
        try:
            probabilities = self.language_model.predict_proba([text])[0]
            max_prob = np.max(probabilities)
            
            if max_prob < self.confidence_threshold:
                print(f"[DEBUG] Low language confidence ({max_prob:.2f} < {self.confidence_threshold}). Defaulting to 'en'.")
                return "en"
            
            class_index = np.argmax(probabilities)
            return self.language_model.classes_[class_index]
        
        except Exception as e:
            print(f"[ERROR] language detection failed: {e}")
            return "en"

    def detect_emotion(self, text: str) -> str:
        try:
            result = self.emotion_model(text)[0]
            return result["label"]
        except Exception as e:
            print(f"[ERROR] emotion detection failed: {e}")
            return "neutral"