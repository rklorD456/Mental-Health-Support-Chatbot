# Final Project

This project is a small NLP prototype for intent classification in a mental-health style chat system.
It was built from the notebooks in the `Notebooks/` folder.

## What is included

- `intent_classifier.py`: classifies user text into one of these intents:
	- `greeting`
	- `goodbye`
	- `gratitude`
	- `asking_mental_health_question`
	- `out_of_scope`
- `main.py`: simple starter script.
- `models/`: saved models created from the notebook experiments.
- `Notebooks/`: notebooks used to train and test the language and emotion models.

## Requirements

- Python 3.10 or later
- A `GROQ_API_KEY` environment variable

## Install

```bash
uv sync
```

If you use the project file, install the declared dependencies with your preferred tool.

## Run

You can test the intent classifier from the command line:

```bash
uv run intent_classifier.py "I have been feeling really overwhelmed and stressed out lately."
```

Or run the default test message:

```bash
uv run intent_classifier.py
```

## Notes

- The classifier uses Groq's OpenAI-compatible API.
- The script expects the API to return valid JSON with a single `intent` field.
- The project models were built from the notebooks in this repository.
- This README describes the project state so far and can be expanded later.
