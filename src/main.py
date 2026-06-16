"""Main FastAPI application entry point.

All model/client initialisation happens in the lifespan context and is stored
on ``AppState``.  Downstream modules receive the state via dependency injection
instead of importing ``src.main`` directly.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from contextlib import asynccontextmanager

import joblib
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline

from src.config import get_settings
from src.core.schemas import ChatRequest, ChatResponse
from src.database.qdrant import get_qdrant_client

from src.rag.embedder import SentenceTransformerEmbedder
from src.rag.retriever import HybridQdrantRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.pipeline import RAGPipeline

from src.services.llm import get_llm_client
from src.services.classifier import ClassifierService
from src.services.translator import TranslatorService

settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))


class AppState:
    rag_pipeline: RAGPipeline
    classifier: ClassifierService
    translator: TranslatorService
    llm_client: object  # openai.OpenAI


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and clients once at startup, yield, then shut down."""

    if not settings.model_used_api or not settings.model_used_base_url:
        raise ValueError("Model API Key or Base URL not found in environment variables.")

    print("Initializing LLM client...")
    state.llm_client = get_llm_client()

    print("Connecting to Qdrant...")
    qdrant_client = get_qdrant_client()

    print("Loading embedding model...")
    embedder = SentenceTransformerEmbedder(settings.embedding_model_name)

    print("Initializing hybrid retriever...")
    retriever = HybridQdrantRetriever(qdrant_client, embedder, settings.collection_name)

    print("Loading cross-encoder reranking model...")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Setting up RAG pipeline...")
    state.rag_pipeline = RAGPipeline(retriever, reranker, state.llm_client, settings)

    print("Loading language detection model...")
    state.translator = TranslatorService(llm_client=state.llm_client, model_name=settings.model_used_name)
    
    print("Loading emotion classification model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state.classifier = ClassifierService(
        language_model_path=str(settings.abs_language_model_path),
        emotion_model_path=str(settings.abs_emotion_model_path),
        device=device, #type: ignore
    )

    print("All models loaded. API is ready.\n")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Mental Health Support Chatbot",
    description="RAG-based chatbot for mental health support.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": hasattr(state, "rag_pipeline")}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_message = request.message
    session_id = request.session_id

    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Input message cannot be empty.")

    from src.router import process_chat

    result = process_chat(user_message, session_id=session_id, app_state=state)
    return ChatResponse(**result)