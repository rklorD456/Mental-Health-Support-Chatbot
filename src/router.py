"""Chat request router.
 
Orchestrates language detection, guardrail checks, emotion/intent classification,
RAG response generation, and conversation history management for each incoming
chat message.
"""

import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
 
from src.config import get_settings
from src.services.intent import get_intent
 
settings = get_settings()


DIRECT_RESPONSES = {
    "greeting":    "Hello! I am here to listen. How are you doing today?",
    "goodbye":     "Take care. Remember, there is always support available if you need it.",
    "gratitude":   "You are very welcome. I am glad I could help.",
    "out_of_scope": "I am a mental health assistant. I cannot help with coding, general trivia, or physical medical advice. How can I support your mental well-being today?"
}


# In-memory conversation history .
MAX_HISTORY_TURNS = 10  # Max user-assistant pairs to keep in history
MAX_SESSIONS = 200  # Max unique session histories to keep in memory

# OrderedDict for reliable LRU eviction + Lock for thread safety
conversation_history: OrderedDict[str, list[dict]] = OrderedDict()
_history_lock = threading.Lock()
    
    

def process_chat(user_message: str, session_id: str = "", app_state=None) -> dict:
    """Process a user message end-to-end and return a structured response dict.

    Args:
        user_message: The raw text received from the user.
        session_id:   Optional session identifier for multi-turn history.
        app_state:    The ``AppState`` instance populated during lifespan startup.

    Returns:
        A dict with keys: ``response``, ``intent``, ``emotion``, ``language``.
    """
    
    if app_state is None:
        raise ValueError("app_state must be provided to process_chat().")
    
    
    classifier   = app_state.classifier
    llm_client   = app_state.llm_client
    rag_pipeline = app_state.rag_pipeline
    translator   = app_state.translator
    executor     = app_state.executor  # lifecycle-managed in AppState

    # --- Language detection & optional translation to English ---
    language = classifier.detect_language(user_message).lower()

    if language != "en":
        user_message = translator.to_english(user_message)
        print(f"[DEBUG] Translated input from {language} to English.")


    # --- RUN SECURITY GUARDRAILS ---
    if not app_state.guardrail.is_safe(user_message):
        # Hash the payload instead of logging raw text to prevent log injection
        payload_hash = hashlib.sha256(user_message.encode()).hexdigest()[:16]
        print(f"[SECURITY ALERT] Prompt injection blocked. hash={payload_hash}")
        
        rejection_text = "I cannot fulfill this request. I am here exclusively to support your mental well-being."
        if language != "en":
            rejection_text = translator.from_english(rejection_text, target_language=language)
 
        return {
            "response": rejection_text,
            "intent":   "out_of_scope",
            "emotion":  "neutral",
            "language": language,   
        }
        
    # --- Run Emotion detection & Intent classification in parallel ---
    future_emotion = executor.submit(classifier.detect_emotion, user_message)
    future_intent = executor.submit(get_intent, user_message, llm_client=llm_client)
    
    # Collect the results as they finish
    try:
        emotion = future_emotion.result(timeout=10)
    except Exception as e:
        print(f"[WARNING] Emotion detection failed ({e}); defaulting to 'neutral'.")
        emotion = "neutral"
 
    try:
        intent = future_intent.result(timeout=10)
    except Exception as e:
        print(f"[WARNING] Intent classification failed ({e}); defaulting to 'out_of_scope'.")
        intent = "out_of_scope"
 
    print(f"[DEBUG] lang={language} | emotion={emotion} | intent={intent}")
    print(f"[DEBUG] message='{user_message}'")

    # ---  Retrieve conversation history ---
    with _history_lock:
        history = list(conversation_history.get(session_id, [])) if session_id else []


    # ---  Generate response ---
    if intent == "asking_mental_health_question":
        response_text = rag_pipeline.generate_response(
            user_message, 
            emotion=emotion, 
            history=history)
    else:
        response_text = DIRECT_RESPONSES.get(intent, DIRECT_RESPONSES["out_of_scope"])

    # ---  Persist conversation history ---
    if session_id:
        with _history_lock:
            if len(conversation_history) >= MAX_SESSIONS and session_id not in conversation_history:
                conversation_history.popitem(last=False)  # evict oldest
 
            if session_id not in conversation_history:
                conversation_history[session_id] = []
 
            conversation_history[session_id].append({"role": "user",      "content": user_message})
            conversation_history[session_id].append({"role": "assistant", "content": response_text})
            conversation_history[session_id] = conversation_history[session_id][-(MAX_HISTORY_TURNS * 2):]
            conversation_history.move_to_end(session_id)
 
    # --- Translate response back to original language if needed ---
    if language != "en":
        response_text = translator.from_english(response_text, target_language=language)
        print(f"[DEBUG] Translated response from English to {language}.")
 
    return {
        "response": response_text,
        "intent":   intent,
        "emotion":  emotion,
        "language": language,
    }
 