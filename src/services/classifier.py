import joblib
from transformers import pipeline

class ClassifierService:
    def __init__(self, language_model_path: str, emotion_model_path: str, device: int):
        self.language_model = joblib.load(language_model_path)
        self.emotion_model = pipeline(
            "text-classification",
            model=emotion_model_path,
            tokenizer=emotion_model_path,
            device=device,
            truncation=True,
            max_length=512
        )

    def detect_language(self, text: str) -> str:
        try:
            return self.language_model.predict([text])[0]
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