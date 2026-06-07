"""
Centralized application configuration.

All settings are loaded from environment variables (with .env fallback)
using Pydantic BaseSettings. Paths are resolved relative to the project
root so the app works on any machine.
"""

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings

# ── Project root (two levels up from this file: app/config.py → project root) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── API Keys ──────────────────────────────────────────────────────────
    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str

    # ── Groq / LLM ───────────────────────────────────────────────────────
    groq_base_url: str = "https://api.groq.com/openai/v1"
    intent_model: str = "openai/gpt-oss-20b"
    translation_model: str = "openai/gpt-oss-20b"
    rag_model: str = "openai/gpt-oss-120b"
    intent_temperature: float = 0.0
    translation_temperature: float = 0.0
    back_translation_temperature: float = 0.3
    rag_temperature: float = 0.3

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_collection_name: str = "mental_health_knowledge_base"
    qdrant_search_limit: int = 3

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"

    # ── Local model paths (relative to project root) ──────────────────────
    language_model_path: str = "models/language_detector_v1.1.joblib"
    emotion_model_path: str = "models/distilbert-emotion-final"

    # ── Misc ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ── Computed helpers ──────────────────────────────────────────────────
    @property
    def abs_language_model_path(self) -> Path:
        return PROJECT_ROOT / self.language_model_path

    @property
    def abs_emotion_model_path(self) -> Path:
        return PROJECT_ROOT / self.emotion_model_path

    @property
    def templates_dir(self) -> Path:
        return PROJECT_ROOT / "templates"

    @property
    def static_dir(self) -> Path:
        return PROJECT_ROOT / "app" / "static"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
