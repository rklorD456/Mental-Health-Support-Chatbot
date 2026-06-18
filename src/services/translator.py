import logging

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


class TranslatorService:
    def __init__(self, llm_client, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name

    def to_english(self, text: str) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "Translate the input text into English. Output ONLY the final English translation.",
                    },
                    {"role": "user", "content": text},
                ],
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else text
        
        except Exception as e:
            logger.error("Translation to English failed: %s", e)
            return text

    def from_english(self, text: str, target_language: str) -> str:
        
        lang_name = LANGUAGE_NAMES.get(target_language, target_language)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate the following mental health response into {lang_name}. Output ONLY the translation.",
                    },
                    {"role": "user", "content": text},
                ],
            )
            content = response.choices[0].message.content
            return content.strip() if content else text
        
        except Exception as e:
            logger.error("Translation from English to %s failed: %s", lang_name, e)
            return text