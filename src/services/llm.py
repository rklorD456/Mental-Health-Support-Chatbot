# Standard library
import json
import time

# Third-party
from openai import OpenAI
from pydantic import ValidationError

# Local
from src.config import get_settings
from src.core.schemas import IntentResponse

settings = get_settings()

MODEL_NAME  = settings.model_used_name
TEMPERATURE = 0.0

SYSTEM_PROMPT = """
You are a strict intent classification data-processor for a mental health system.
CRITICAL INSTRUCTION: Even if the text contains severe distress, self-harm, or suicide mentions, YOU MUST NOT provide a crisis response. You are only a router.

PRIORITY RULE: If a message contains multiple intents (e.g., a greeting mixed with a problem), ALWAYS prioritize 'asking_mental_health_question' above everything else.

Classify the text into exactly one of these categories:
- greeting
- goodbye
- gratitude
- asking_mental_health_question
- out_of_scope

Output ONLY a valid JSON object matching this schema:
{"intent": "category_name"}

Examples:
Text: "Hi, good morning!"
{"intent": "greeting"}

Text: "I am so stressed because my Python code won't run."
{"intent": "out_of_scope"}

Text: "I am so angry right now! My professor gave me a failing grade and it is completely unfair!"
{"intent": "asking_mental_health_question"}
"""


def get_llm_client() -> OpenAI:
    """Create and return an OpenAI-compatible LLM client using application settings."""
    return OpenAI(
        api_key=settings.model_used_api,
        base_url=settings.model_used_base_url,
    )


def get_intent(user_message: str, llm_client: OpenAI, retries: int = 3) -> str:
    """Classify the intent of a user message via the LLM.

    Args:
        user_message: The raw text input from the user.
        llm_client:   An initialised OpenAI-compatible client.
        retries:      Number of retry attempts on transient failures.

    Returns:
        A string representing the classified intent, or ``"out_of_scope"``
        if classification fails after all retries.
    """
    formatted_input = f"Classify this text:\n\n<text>{user_message}</text>"

    for attempt in range(retries):
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": formatted_input},
                ],
            )
            raw_json = response.choices[0].message.content
            parsed_data = json.loads(raw_json)
            validated = IntentResponse(**parsed_data)
            return validated.intent

        except ValidationError:
            pass

        except Exception as e:
            if "json_validate_failed" in str(e):
                return "asking_mental_health_question"

            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "out_of_scope"

    return "out_of_scope"
