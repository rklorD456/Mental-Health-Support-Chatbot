# Local
from src.config import get_settings
from src.services.llm import get_intent
from src.services.translator import TranslatorService

settings = get_settings()

DIRECT_RESPONSES = {
    "greeting":    "Hello! I am here to listen. How are you doing today?",
    "goodbye":     "Take care. Remember, there is always support available if you need it.",
    "gratitude":   "You are very welcome. I am glad I could help.",
    "out_of_scope": "I am a mental health assistant. I cannot help with coding, general trivia, or physical medical advice. How can I support your mental well-being today?"
}

# In-memory conversation history keyed by session_id.
# Each value is a list of {"role": "user"|"assistant", "content": str} dicts.
MAX_HISTORY_TURNS = 10  # keep the last N exchanges (user + assistant = 2 msgs each)
conversation_history: dict[str, list[dict]] = {}


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

    translator = TranslatorService(llm_client=llm_client, model_name=settings.model_used_name)

    # --- Step 1: Language detection & optional translation to English ---
    language = classifier.detect_language(user_message).lower()

    if language != "en":
        user_message = translator.to_english(user_message)
        print(f"[DEBUG] Translated input from {language} to English.")

    # --- Step 2: Emotion detection ---
    emotion = classifier.detect_emotion(user_message)

    # --- Step 3: Intent classification ---
    intent = get_intent(user_message, llm_client=llm_client)

    print(f"[DEBUG] lang={language} | emotion={emotion} | intent={intent}")
    print(f"[DEBUG] message='{user_message}'")

    # --- Step 4: Retrieve conversation history ---
    history = conversation_history.get(session_id, []) if session_id else []

    # --- Step 5: Generate response ---
    if intent == "asking_mental_health_question":
        response_text = rag_pipeline.generate_response(user_message, emotion=emotion, history=history)
    else:
        response_text = DIRECT_RESPONSES.get(intent, DIRECT_RESPONSES["out_of_scope"])

    # --- Step 6: Persist conversation history ---
    if session_id:
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        conversation_history[session_id].append({"role": "user",      "content": user_message})
        conversation_history[session_id].append({"role": "assistant", "content": response_text})
        # Trim to keep only the last N turns (each turn = 2 messages)
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
