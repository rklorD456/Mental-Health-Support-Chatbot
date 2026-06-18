# tests/

Test scripts for the chatbot.

## Files

| File | Purpose |
|------|---------|
| `test_intent.py` | Evaluation script for the LLM-based intent classifier. Runs 70+ test cases against the live API and reports accuracy. |

## Running Tests

```bash
uv run python -m tests.test_intent
```

> **Note:** This test makes live LLM API calls and includes a 4-second delay between cases to avoid rate limits. A full run takes ~5 minutes.
