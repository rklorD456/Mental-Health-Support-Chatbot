# src/services/

Service modules for NLP tasks: classification, translation, intent detection, and safety.

## Files

| File | Purpose |
|------|---------|
| `classifier.py` | `ClassifierService` — language detection (scikit-learn joblib model) and emotion detection (fine-tuned DistilBERT via HuggingFace `pipeline`) |
| `intent.py` | `get_intent()` — classifies user messages into one of 5 intent categories using an LLM call with structured JSON output |
| `translator.py` | `TranslatorService` — translates user input to English (for processing) and responses back to the user's language (via LLM) |
| `guardrail.py` | `GuardrailService` — detects prompt injection, jailbreak attempts, and suspicious input patterns using regex + unicode normalization |
