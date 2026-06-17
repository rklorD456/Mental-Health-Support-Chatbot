"""Main FastAPI application entry point.

All model/client initialisation happens in the lifespan context and is stored
on ``AppState``.  Downstream modules receive the state via dependency injection
instead of importing ``src.main`` directly.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"


import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import asyncio

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

from src.services.intent import get_llm_client
from src.services.classifier import ClassifierService
from src.services.translator import TranslatorService
from src.services.guardrail import GuardrailService
from src.router import process_chat


logger = logging.getLogger(__name__)
settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))

 
##--------------------------------------------------
## --- Application state ---
##--------------------------------------------------
class AppState:
    """Holds every shared resource initialised at startup.
 
    All attributes are pre-set to ``None`` so that a partially-initialised
    instance fails with a clear ``AttributeError`` rather than an opaque
    ``AttributeError: 'AppState' object has no attribute '...'``.
    """
    def __init__(self):
        self.rag_pipeline : RAGPipeline | None = None
        self.classifier : ClassifierService | None = None
        self.translator : TranslatorService | None = None
        self.guardrail : GuardrailService | None = None
        self.llm_client : object | None = None
        self.executor : ThreadPoolExecutor | None = None

    def is_ready(self) -> bool:
        """Check if all components are initialised."""
        return all([
            self.rag_pipeline,
            self.classifier,
            self.translator,
            self.guardrail,
            self.llm_client,
            self.executor,
        ])
        

state = AppState()



##--------------------------------------------------
## --- lifespan management ---
##--------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and clients once at startup, yield, then shut down cleanly."""

    if not settings.model_used_api or not settings.model_used_base_url:
        raise ValueError("Model API Key or Base URL not found in environment variables.")

    logger.info("Initializing LLM client...")
    state.llm_client = get_llm_client()

    logger.info("Connecting to Qdrant...")
    qdrant_client = get_qdrant_client()

    logger.info("Loading embedding model...")
    embedder = SentenceTransformerEmbedder(settings.embedding_model_name)

    logger.info("Initializing hybrid retriever...")
    retriever = HybridQdrantRetriever(qdrant_client, embedder, settings.collection_name)

    logger.info("Loading cross-encoder reranking model...")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    logger.info("Setting up RAG pipeline...")
    state.rag_pipeline = RAGPipeline(retriever, reranker, state.llm_client, settings)

    logger.info("Loading language detection model...")
    state.translator = TranslatorService(llm_client=state.llm_client, model_name=settings.model_used_name)
    
    logger.info("Loading emotion & language classification models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state.classifier = ClassifierService(
        language_model_path=str(settings.abs_language_model_path),
        emotion_model_path=str(settings.abs_emotion_model_path),
        device=device,          # type: ignore[arg-type]
        confidence_threshold=0.65,
    )
    
    logger.info("Setting up safety guardrails...")
    state.guardrail = GuardrailService()
    
    # Executor is created here so it participates in the application lifecycle
    # and is shut down gracefully when the server stops.
    logger.info("Starting thread-pool executor...")
    state.executor = ThreadPoolExecutor(max_workers=4)
 
    logger.info("All models loaded. API is ready.")
    yield

    ## Clean shutdown of resources happens here after the server stops accepting requests.
    logger.info("Shutting down executor...")
    state.executor.shutdown(wait=True)
    logger.info("Shutdown complete.")


## --- FastAPI app and routes ---
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
    return {"status": "ok" if state.is_ready() else "inittializing",
            "pipeline_loaded": hasattr(state, "rag_pipeline"),
            "classifier_loaded": hasattr(state, "classifier"),
            "translator_loaded": hasattr(state, "translator"),
            "guardrail_loaded": hasattr(state, "guardrail")}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    
    user_message = request.message
    session_id = request.session_id

    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Input message cannot be empty.")
 
    if not state.is_ready():
        raise HTTPException(status_code=503, detail="Service is still initializing. Please retry shortly.")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        state.executor,
        lambda: process_chat(user_message, session_id=session_id, app_state=state),
    )
    
    return ChatResponse(**result)

