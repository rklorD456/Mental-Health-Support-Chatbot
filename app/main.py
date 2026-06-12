import joblib
from contextlib import asynccontextmanager

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException, Request

from transformers import pipeline
import torch
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse

settings = get_settings()
templates = Jinja2Templates(directory=str(settings.templates_dir))

models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan function to load all models and clients once at startup."""

    if not settings.model_used_api or not settings.model_used_base_url:
        raise ValueError("Model API Key or Base URL not found in environment variables.")
    
    # LLM client (shared across intent classification, RAG, and translation)
    print("Initializing LLM client...")
    models["llm_client"] = OpenAI(
        api_key=settings.model_used_api,
        base_url=settings.model_used_base_url,
    )
    
    # Qdrant vector database client
    print("Connecting to Qdrant...")
    models["qdrant_client"] = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    
    # Embedding model for RAG retrieval
    print("Loading embedding model...")
    models["embedding"] = SentenceTransformer(settings.embedding_model_name)
    
    # Language detection model
    print("Loading language detection model...")
    models["language"] = joblib.load(settings.abs_language_model_path)
 
    # Emotion classification model
    print("Loading emotion classification model...")
    device = 0 if torch.cuda.is_available() else "cpu"
    emotion_model_path = str(settings.abs_emotion_model_path)
    models["emotion"] = pipeline(
        "text-classification",
        model=emotion_model_path,
        tokenizer=emotion_model_path,
        device=device,
        truncation=True,
        max_length=512
    )
    
    # Cross-encoder model for reranking in hybrid retrieval
    print("Loading cross-encoder reranking model...")
    models["cross_encoder"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    print("All models loaded. API is ready.\n")
    
    yield
    # Shutdown: release resources
    models.clear()
    print("Models released. Shutting down.")
    
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
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_message = request.message
    
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Input message cannot be empty.")
    
    from app.router import process_chat
    result = process_chat(user_message)
        
    return ChatResponse(**result)