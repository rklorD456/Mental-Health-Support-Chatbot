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
    lightning_api_key = os.getenv("LIGHTNING_API_KEY", "")
    lightning_base_url = os.getenv("LIGHTNING_BASE_URL", "https://lightning.ai/api/v1")
    
    ## what model i wan to use for RAG
    model_used_api = lightning_api_key
    model_used_base_url = lightning_base_url
    
    model_used_name = "openai/gpt-4o"
    model_used_name_intent = "lightning-ai/gpt-oss-20b"
    model_used_name_translation = "lightning-ai/gpt-oss-20b"
    
    ## for embedding
    embedding_model_name = "BAAI/bge-base-en-v1.5"
    collection_name = "mental_health_knowledge_base"
    model_used_temperature = 0.3
    
    
    abs_language_model_path = PROJECT_ROOT / "models" / "language_detector_v1.1.joblib"
    abs_emotion_model_path = PROJECT_ROOT / "models" / "distilbert-emotion-final"
    templates_dir = PROJECT_ROOT / "templates"

_settings = Settings()

def get_settings():
    """Return a cached singleton of the application settings."""
    return _settings
