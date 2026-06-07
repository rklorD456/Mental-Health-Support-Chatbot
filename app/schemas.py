"""
Pydantic request / response schemas for the API.
"""

from typing import Literal
from pydantic import BaseModel, Field


# ── Chat Schemas ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming user message."""
    message: str = Field(..., min_length=1, description="User message text")


class ChatResponse(BaseModel):
    """Structured chatbot response returned to the client."""
    response: str
    intent: str
    emotion: str
    language: str


# ── Internal Schemas ──────────────────────────────────────────────────────────

class IntentResponse(BaseModel):
    """Expected JSON structure from the intent-classification LLM call."""
    intent: Literal[
        "greeting",
        "goodbye",
        "gratitude",
        "asking_mental_health_question",
        "out_of_scope",
    ]
