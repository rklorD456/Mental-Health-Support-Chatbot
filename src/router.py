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

# In-memory conversation history keyed by session_id.
# Each value is a list of {"role": "user"|"assistant", "content": str} dicts.
MAX_HISTORY_TURNS = 10  # Max user-assistant pairs to keep in history
conversation_history: dict[str, list[dict]] = {}

executor = ThreadPoolExecutor(max_workers=2)

def process_chat(user_message: str, session_id: str = "", app_state=None) -> dict:
    """Process a user message end-to-end and return a structured response dict.

    Args:
        user_message: The raw text received from the user.
        session_id:   Optional session identifier for multi-turn history.
        app_state:    The ``AppState`` instance populated during lifespan startup.

    Returns:
        A dict with keys: ``response``, ``intent``, ``emotion``, ``language``.
    """
    classifier  = app_state.classifier
    llm_client  = app_state.llm_client
    rag_pipeline = app_state.rag_pipeline
    translator = app_state.translator

    # --- Language detection & optional translation to English ---
    language = classifier.detect_language(user_message).lower()

    if language != "en":
        user_message = translator.to_english(user_message)
        print(f"[DEBUG] Translated input from {language} to English.")


    # --- Run Emotion detection & Intent classification in parallel ---
    future_emotion = executor.submit(classifier.detect_emotion, user_message)
    future_intent = executor.submit(get_intent, user_message, llm_client=llm_client)
    
    # Collect the results as they finish
    emotion = future_emotion.result()
    intent = future_intent.result()
    
    print(f"[DEBUG] lang={language} | emotion={emotion} | intent={intent}")
    print(f"[DEBUG] message='{user_message}'")

    # ---  Retrieve conversation history ---
    history = conversation_history.get(session_id, []) if session_id else []

    # ---  Generate response ---
    if intent == "asking_mental_health_question":
        response_text = rag_pipeline.generate_response(user_message, 
                                                       emotion=emotion, 
                                                       history=history)
    else:
        response_text = DIRECT_RESPONSES.get(intent, DIRECT_RESPONSES["out_of_scope"])

    # ---  Persist conversation history ---
    if session_id:
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        conversation_history[session_id].append({"role": "user", "content": user_message})
        conversation_history[session_id].append({"role": "assistant", "content": response_text})
        conversation_history[session_id] = conversation_history[session_id][-(MAX_HISTORY_TURNS * 2):]

    # --- Step 7: Translate response back to original language if needed ---
    if language != "en":
        response_text = translator.from_english(response_text, target_language=language)
        print(f"[DEBUG] Translated response from English to {language}.")

    return {
        "response": response_text,
        "intent":   intent,
        "emotion":  emotion,
        "language": language,
    }
