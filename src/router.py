"""Chat request router.
 
Orchestrates language detection, guardrail checks, emotion/intent classification,
fast keyword matching, RAG response generation, and conversation history management.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
 
from src.config import get_settings
from src.services.intent import get_intent
from src.core.responses import detect_intent_from_keywords, get_dynamic_response
 
settings = get_settings()
logger = logging.getLogger(__name__)


# In-memory conversation history .
MAX_HISTORY_TURNS = 10  # Max user-assistant pairs to keep in history
MAX_SESSIONS = 200  # Max unique session histories to keep in memory

# OrderedDict for reliable LRU eviction + Lock for thread safety
conversation_history: OrderedDict[str, list[dict]] = OrderedDict()
_history_lock = threading.Lock()
    
def is_meaningless_input(text: str) -> bool:
    """Fast check to drop low-effort or invalid inputs before processing."""
    clean = text.strip()
    
    # Empty string or whitespace only
    if not clean:
        return True
    
    # Only numbers (e.g., "123", "0")
    if clean.isnumeric():
        return True
        
    # Only punctuation (e.g., "!!!", "...")
    if all(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~" for char in clean):
        return True
        
    # Only one character (e.g., "?", "a", "x")
    if len(clean) == 1:
        return True
        
    # Exactly two identical characters (e.g., "aa", "!!", "..")
    if len(clean) == 2 and clean[0] == clean[1]:
        return True
        
    return False

def _update_history(session_id: str, user_message: str, response_text: str):
    """Thread-safe helper to append messages and enforce LRU memory limits."""
    if not session_id:
        return
        
    with _history_lock:
        # Evict the oldest session if we hit the memory cap
        if len(conversation_history) >= MAX_SESSIONS and session_id not in conversation_history:
            conversation_history.popitem(last=False) 

        if session_id not in conversation_history:
            conversation_history[session_id] = []

        conversation_history[session_id].append({"role": "user",      "content": user_message})
        conversation_history[session_id].append({"role": "assistant", "content": response_text})
        
        # Enforce turn limits
        conversation_history[session_id] = conversation_history[session_id][-(MAX_HISTORY_TURNS * 2):]
        # Move to the end to mark it as recently used (LRU logic)
        conversation_history.move_to_end(session_id)    

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
    client       = app_state.client
    rag_pipeline = app_state.rag_pipeline
    translator   = app_state.translator
    executor     = app_state.executor  # lifecycle-managed in AppState

    original_message = user_message
    
    # --- FAST PRE-PROCESSING (Low-Effort Filter) ---
    if is_meaningless_input(user_message):
        logger.debug("Blocked meaningless input: '%s'", user_message)
        fallback_response = "I didn't quite catch that. Could you please tell me a bit more about what's on your mind?"
        return {
            "response": fallback_response,
            "intent":   "out_of_scope",
            "emotion":  "neutral",
            "language": "en", # Defaulting to English for gibberish
        }
    
    
    # --- Language detection & optional translation to English ---
    language = classifier.detect_language(user_message).lower()
    
    # --- 3. FAST KEYWORD MATCH (O(1) dictionary bypass) ---
    fast_intent = detect_intent_from_keywords(original_message)
    if fast_intent:
        logger.debug("Quick match triggered -> %s", fast_intent)
        response_text = get_dynamic_response(fast_intent, language)
        
        _update_history(session_id, original_message, response_text)
        return {
            "response": response_text,
            "intent":   fast_intent,
            "emotion":  "neutral",
            "language": language,
        }
        
    if language != "en":
        user_message = translator.to_english(user_message)
        logger.debug("Translated input from %s to English.", language)


    # --- RUN SECURITY GUARDRAILS ---
    if not app_state.guardrail.is_safe(user_message):
        # Hash the payload instead of logging raw text to prevent log injection
        payload_hash = hashlib.sha256(user_message.encode()).hexdigest()[:16]
        logger.warning("Prompt injection blocked. hash=%s", payload_hash)
        
        rejection_text = get_dynamic_response("out_of_scope", language)
 
        return {
            "response": rejection_text,
            "intent":   "out_of_scope",
            "emotion":  "neutral",
            "language": language,   
        }
        
    # --- Run Emotion detection & Intent classification in parallel ---
    future_emotion = executor.submit(classifier.detect_emotion, user_message)
    future_intent = executor.submit(get_intent, user_message, llm_client=client)
    
    # Collect the results as they finish
    try:
        emotion = future_emotion.result(timeout=10)
    except Exception as e:
        logger.warning("Emotion detection failed (%s); defaulting to 'neutral'.", e)
        emotion = "neutral"
 
    try:
        intent = future_intent.result(timeout=10)
    except Exception as e:
        logger.warning("Intent classification failed (%s); defaulting to 'out_of_scope'.", e)
        intent = "out_of_scope"
 
    logger.debug("lang=%s | emotion=%s | intent=%s", language, emotion, intent)
    logger.debug("message='%s'", user_message)

    # ---  Retrieve conversation history ---
    with _history_lock:
        history = list(conversation_history.get(session_id, [])) if session_id else []


    # ---  Generate response ---
    if intent == "asking_mental_health_question":
        response_text = rag_pipeline.generate_response(
            user_message, 
            emotion=emotion, 
            history=history)
        
        if language != "en":
                    response_text = translator.from_english(response_text, target_language=language)
                    logger.debug("Translated RAG response from English to %s.", language)
    else:
        # If it's a normal intent that the LLM caught (but missed the fast keyword filter),
        # pull a native dynamic response so we don't have to translate it.
        response_text = get_dynamic_response(intent, language)

    # ---  Persist conversation history ---
    _update_history(session_id, original_message, response_text)
 
    return {
        "response": response_text,
        "intent":   intent,
        "emotion":  emotion,
        "language": language,
    }
 