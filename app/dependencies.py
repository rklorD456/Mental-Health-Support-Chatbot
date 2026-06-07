"""
Shared dependency singletons.

All heavy clients and ML models are created here **once** and imported
by the service modules. This eliminates the duplicated client creation
that was previously spread across main.py, intent_classifier.py, and
rag_retriever.py.
"""

import logging
from typing import Any

import joblib
import torch
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ── Groq (OpenAI-compatible) client ──────────────────────────────────────────
groq_client = OpenAI(
    api_key=settings.groq_api_key,
    base_url=settings.groq_base_url,
)

# ── Qdrant vector-database client ────────────────────────────────────────────
qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)

# ── Embedding model (sentence-transformers) ──────────────────────────────────
embedding_model = SentenceTransformer(settings.embedding_model_name)


# ── ML model registry (populated during FastAPI lifespan) ────────────────────
#    language & emotion models are large and should only be loaded once,
#    so we expose a mutable dict that the lifespan handler fills.
ml_models: dict[str, Any] = {}


def load_ml_models() -> None:
    """Load the language-detection and emotion-classification models."""
    logger.info("Loading language detection model from %s …", settings.abs_language_model_path)
    ml_models["language"] = joblib.load(settings.abs_language_model_path)

    device = 0 if torch.cuda.is_available() else "cpu"
    logger.info("Loading emotion classification model from %s (device=%s) …",
                settings.abs_emotion_model_path, device)
    ml_models["emotion"] = hf_pipeline(
        "text-classification",
        model=str(settings.abs_emotion_model_path),
        tokenizer=str(settings.abs_emotion_model_path),
        device=device,
        truncation=True,
        max_length=512,
    )
    logger.info("All ML models loaded successfully.")


def unload_ml_models() -> None:
    """Release ML model resources."""
    ml_models.clear()
    logger.info("ML models released.")
