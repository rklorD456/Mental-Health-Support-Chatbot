import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings:
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    embedding_model_name = "BAAI/bge-base-en-v1.5"
    
    abs_language_model_path = PROJECT_ROOT / "models" / "language_detector_v1.1.joblib"
    abs_emotion_model_path = PROJECT_ROOT / "models" / "distilbert-emotion-final"
    templates_dir = PROJECT_ROOT / "templates"

_settings = Settings()

def get_settings():
    """Return a cached singleton of the application settings."""
    return _settings
