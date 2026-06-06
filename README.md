# Final Project

This project is a small NLP prototype for intent classification in a mental-health style chat system.
It was built from the notebooks in the `Notebooks/` folder.
It also includes a data ingestion script that builds a Qdrant knowledge base from a mental-health conversation dataset.

## What is included

- `intent_classifier.py`: classifies user text into one of these intents:
	- `greeting`
	- `goodbye`
	- `gratitude`
	- `asking_mental_health_question`
	- `out_of_scope`
- `ingest_data.py`: downloads a mental-health counseling dataset and uploads it to Qdrant as vectors.
- `main.py`: simple starter script.
- `models/`: saved models created from the notebook experiments.
- `Notebooks/`: notebooks used to train and test the language and emotion models.

## Requirements

- Python 3.10 or later
- A `GROQ_API_KEY` environment variable
- A `QDRANT_URL` environment variable
- A `QDRANT_API_KEY` environment variable

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

## Ingest Data

The `ingest_data.py` script creates or updates the `mental_health_knowledge_base` collection in Qdrant.
It uses the `Amod/mental_health_counseling_conversations` dataset and the `BAAI/bge-base-en-v1.5` sentence embedding model.

Run it when you want to build the vector database used for retrieval:

```bash
uv run ingest_data.py
```

## Notes

- The classifier uses Groq's OpenAI-compatible API.
- The script expects the API to return valid JSON with a single `intent` field.
- The project models were built from the notebooks in this repository.
- The ingestion script stores each conversation response as an embedding and saves the related question and answer as metadata.
- This README describes the project state so far and can be expanded later.
