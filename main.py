import os
import joblib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from transformers import pipeline
import torch
from intent_classifier import get_intent
from rag_retriever import get_rag_response

load_dotenv()

models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan function to load models at startup and clean up if necessary on shutdown."""
    
    print("Loading language detection model...")
    models["language"] = joblib.load("./models/language_detector_v1.1.joblib")
 
    print("Loading emotion classification model...")

    device = 0 if torch.cuda.is_available() else "cpu"
    
    models["emotion"] = pipeline(
        "text-classification",
        model="./models/distilbert-emotion-final",
        tokenizer="./models/distilbert-emotion-final",
        device=device,
        truncation=True,
        max_length=512
    )
    
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


# data models
class ChatRequest(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    response: str
    intent: str
    emotion: str
    language: str

def detect_language(text: str) -> str:
    """Detects the language of the input text using a pre-trained model.
    
    Args:
        text (str): The input text whose language needs to be detected.
        
    Returns:
        str: The detected language label (e.g., "en", "es", etc.).
    """
    try:
        language = models["language"].predict([text])[0]
        return language    
    except Exception as e:
        print(f"[ERROR] Language detection failed: {e}")
        return "en"


def detect_emotion(text: str) -> str:
    """Detects the emotion expressed in the input text using a fine-tuned transformer model.
    
    Args:
        text (str): The input text whose emotion needs to be detected.
    Returns:
        str: The detected emotion label (e.g., "joy", "sadness", "anger", etc.).
    """
    try:
        result = models["emotion"](text)[0]
        emotion = result["label"]
        return emotion
    except Exception as e:
        print(f"[ERROR] Emotion detection failed: {e}")
        return "neutral"
    
    
# for non-rag intents 
DIRECT_RESPONSES = {
    "greeting": "Hello! I am here to listen. How are you doing today?",
    "goodbye": "Take care. Remember, there is always support available if you need it.",
    "gratitude": "You are very welcome. I am glad I could help.",
    "out_of_scope": "I am a mental health assistant. I cannot help with coding, general trivia, or physical medical advice. How can I support your mental well-being today?"
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health Support Chatbot API."}

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_message = request.message
    
    if not user_message.strip():
        raise HTTPException(status_code=400, detail="Input message cannot be empty.")
    
    
    # Detect Language
    language = detect_language(user_message)
    
    # Detect Emotion
    emotion = detect_emotion(user_message)
    
    # Classify Intent
    intent = get_intent(user_message)
    
    print(f"[DEBUG] lang={language} | emotion={emotion} | intent={intent}")
    print(f"[DEBUG] message='{user_message[:80]}'")
    
    
    # Generate Response
    if intent == "asking_mental_health_question":
        response_text = get_rag_response(user_message, emotion=emotion)
    else:
        response_text = DIRECT_RESPONSES.get(intent, DIRECT_RESPONSES["out_of_scope"])
    
    return ChatResponse(
        response=response_text,
        intent=intent,
        emotion=emotion,
        language=language
    )
        