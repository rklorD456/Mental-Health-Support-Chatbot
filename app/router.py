from app.intent_classifier import get_intent
from app.rag_retriever import get_rag_response
import app.main as main


DIRECT_RESPONSES = {
    "greeting": "Hello! I am here to listen. How are you doing today?",
    "goodbye": "Take care. Remember, there is always support available if you need it.",
    "gratitude": "You are very welcome. I am glad I could help.",
    "out_of_scope": "I am a mental health assistant. I cannot help with coding, general trivia, or physical medical advice. How can I support your mental well-being today?"
}


def detect_language(text: str) -> str:
    """Detects the language of the input text using a pre-trained model.
    
    Args:
        text (str): The input text whose language needs to be detected.
        
    Returns:
        str: The detected language label (e.g., "en", "es", etc.).
    """
    try:
        language = main.models["language"].predict([text])[0]
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
        result = main.models["emotion"](text)[0]
        emotion = result["label"]
        return emotion
    except Exception as e:
        print(f"[ERROR] Emotion detection failed: {e}")
        return "neutral"
    

def translate_to_english(text: str, source_lang: str) -> str:
    """Translates text from the source language to English using a translation API .
    
    Args:
        text (str): The input text in the source language.
        source_lang (str): The language code of the input text (e.g., "es" for Spanish).
        
    Returns:
        str: The translated text in English.
    """
    try:
        response = main.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a highly accurate translator. Translate the input text into English. Output ONLY the final English translation. Do not include any notes, quotes, explanations, or extra punctuation."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Translation to English failed: {e}")
        return text
    

def translate_from_english(text: str, target_language: str) -> str:
    """Translates the final English response back to the user's native language code/name."""
    try:
        response = main.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            messages=[
                {"role": "system", "content": f"Translate the following mental health response into the language specified: '{target_language}'. Maintain a warm, highly empathetic, and professional tone. Output ONLY the translation without any notes or metadata."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Translation from English failed: {e}")
        return text   


def process_chat(user_message: str) -> dict:
    language = detect_language(user_message).lower()
    
    if str(language) != 'en':
        user_message = translate_to_english(user_message, source_lang=language)
        print(f"[DEBUG] Translated input from {language} to English.")
        
    emotion = detect_emotion(user_message)
    
    intent = get_intent(user_message)
    
    print(f"[DEBUG] lang={language} | emotion={emotion} | intent={intent}")
    print(f"[DEBUG] message='{user_message[:80]}'")
    
    if intent == "asking_mental_health_question":
        response_text = get_rag_response(user_message, emotion=emotion)
    else:
        response_text = DIRECT_RESPONSES.get(intent, DIRECT_RESPONSES["out_of_scope"])
    
    if str(language) != 'en':
        response_text = translate_from_english(response_text, target_language=language)
        print(f"[DEBUG] Translated response from English to {language}.")
        
    return {
        "response": response_text,
        "intent": intent,
        "emotion": emotion,
        "language": language,
    }
