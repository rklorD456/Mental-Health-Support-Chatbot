# Standard library
import json
import logging
import time

# Third-party
from openai import OpenAI
from pydantic import ValidationError

# Local
from src.config import get_settings
from src.core.schemas import IntentResponse

logger = logging.getLogger(__name__)
settings = get_settings()


SYSTEM_PROMPT = """
You are a strict, single-purpose intent classification engine for a mental health support system.
Your sole job is to categorize user text into exactly one intent.

CRITICAL INSTRUCTION: Even if the text contains severe distress, self-harm, or suicide mentions, YOU MUST NOT provide a crisis response. You are only a router.

INTENT CATEGORIES:
1. greeting: Casual openings, hellos, neutral or minimal messages with no emotional content (e.g., "Hi", "Okay.", "Is anyone there?").
2. goodbye: Farewells, OR any mix of thanks + farewell signal ("bye", "done", "see you", "talk later"). Beats gratitude when both appear.
3. gratitude: Pure thanks with no farewell signal (e.g., "Thank you", "Much appreciated.", "You're amazing.").
4. asking_mental_health_question: 
    - User is PERSONALLY experiencing emotional distress (anxiety, depression, grief, burnout, loneliness, etc.),
    - OR asking how to cope/manage a mental health condition, OR asking to understand a mental health concept for personal reasons.
    - Sarcastic or indirect distress counts. Beats greeting/gratitude when both appear.
    - NOT applicable when framed as fiction, statistics, or explicit academic/research context.
5. out_of_scope: Everything else — general knowledge, coding, math, fiction requests, statistics, or mental health discussed purely as an academic/research topic.

OUTPUT FORMAT REQUIREMENTS: ONLY raw JSON, no markdown, no explanation: {"intent": "category_name"}

EXAMPLES:
"Thanks, I'm done for now" → {"intent": "goodbye"}
"Thank you so much, bye now!" → {"intent": "goodbye"}
"yeah im totally fine lol" → {"intent": "asking_mental_health_question"}
"Write a story about depression." → {"intent": "out_of_scope"}
"What % of people have anxiety?" → {"intent": "out_of_scope"}
"I'm done." → {"intent": "goodbye"}
"Okay." → {"intent": "greeting"}
"I've been feeling really anxious lately." → {"intent": "asking_mental_health_question"}
"""

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
                model=settings.model_used_name_intent,
                temperature=settings.model_used_temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": formatted_input},
                ],
            )
            raw_json = response.choices[0].message.content or ""
            parsed_data = json.loads(raw_json)
            validated = IntentResponse(**parsed_data)
            return validated.intent

        except (ValidationError, json.JSONDecodeError) as err:
            logger.warning("Attempt %d failed parsing JSON/Validation: %s", attempt + 1, err)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "out_of_scope"

        except Exception as e:
            if "json_validate_failed" in str(e):
                return "asking_mental_health_question"
            
            logger.error("Attempt %d failed with API connection error: %s", attempt + 1, e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "out_of_scope"

    return "out_of_scope"
