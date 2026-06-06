import os
import sys
import time
import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

class IntentResponse(BaseModel):
    intent: Literal[
        "greeting",
        "goodbye",
        "gratitude",
        "asking_mental_health_question",
        "out_of_scope"
    ]
    
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


MODEL_NAME    = "openai/gpt-oss-20b"
TEMPERATURE   = 0.0


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

Text: "I have been feeling really overwhelmed and stressed out lately."
{"intent": "asking_mental_health_question"}
"""



def get_intent(user_message: str, retries: int = 3) -> str:
    """
    Takes a user message and returns only the intent string.
    
    inputs:
        - user_message: The raw text input from the user.
        - retries: Number of times to retry the API call in case of failure.
        
    outputs:
        - A string representing the classified intent, or "out_of_scope" if classification fails after retries. 

    """
    formatted_input = f"Classify this text:\n\n<text>{user_message}</text>"
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
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

# Optional: Quick local test block (won't run when imported elsewhere)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_msg = " ".join(sys.argv[1:])
        result = get_intent(test_msg)
        print(f"Message: {test_msg}")
        print(f"Intent:  {result}")
    

    else:
        test_message = "I have been feeling really overwhelmed and stressed out lately."
        result = get_intent(test_message)
        print(f"Message: {test_message}")
        print(f"Intent:  {result}")