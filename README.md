# 🧠 Mental Health Support Chatbot

A RAG-powered chatbot that provides empathetic mental health support. It detects user intent, emotion, and language — then retrieves relevant counseling conversations from a vector database to generate compassionate, context-aware responses.

## How It Works

```
User Message
    │
    ├─ 1. Language Detection (scikit-learn)
    ├─ 2. Fast Keyword Match (local dictionary bypass)
    ├─ 3. Guardrail Check (prompt injection filter)
    │
    ├─ 4. Emotion Detection (DistilBERT)     ← runs in parallel
    ├─ 5. Intent Classification (GPT-4o)     ← runs in parallel
    │
    ├─ 6. RAG Retrieval + Reranking (Qdrant + CrossEncoder)
    └─ 7. Response Generation (GPT-4o, emotion-aware)
         │
         └─ Final Response (translated back if needed)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI |
| LLM | GPT-4o via Lightning AI (OpenAI-compatible) |
| Emotion Detection | DistilBERT (fine-tuned) |
| Language Detection | scikit-learn classifier |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Vector Database | Qdrant Cloud |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Frontend | Jinja2 + vanilla JS |

## Project Structure

```
Mental-Health-Support-Chatbot/
├── main.py                  # Entry point — runs uvicorn
├── pyproject.toml           # Dependencies & build config
├── .env.example             # Template for environment variables
│
├── src/                     # Application source code
│   ├── main.py              #   FastAPI app, lifespan, routes
│   ├── config.py            #   Settings & env var loading
│   ├── router.py            #   Chat orchestration logic
│   ├── ingest_data.py       #   Qdrant data ingestion script
│   ├── core/                #   Schemas, interfaces, keyword responses
│   ├── database/            #   Qdrant client & collection helpers
│   ├── rag/                 #   Embedder, retriever, reranker, pipeline
│   └── services/            #   Classifier, intent, translator, guardrail
│
├── models/                  # Pre-trained model files (gitignored)
├── templates/               # Jinja2 HTML templates
├── static/                  # CSS & JS for the chat UI
├── tests/                   # Test scripts
├── Notebooks/               # Jupyter notebooks (model training)
└── assets/                  # Images for documentation
```

Each folder contains its own `README.md` with more details.

## Setup

### Prerequisites

- Python 3.10 – 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/rklorD456/Mental-Health-Support-Chatbot.git
cd Mental-Health-Support-Chatbot

# Copy and fill in your API keys
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/macOS

# Install dependencies
uv sync
```

### Environment Variables

Create a `.env` file from `.env.example` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `LIGHTNING_API_KEY` | ✅ | Lightning AI API key (for LLM inference) |
| `QDRANT_URL` | ✅ | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | Qdrant Cloud API key |

## Usage

### Start the server

```bash
uv run python main.py
```

Open `http://localhost:8000` for the chat UI, or `http://localhost:8000/docs` for the API docs.

### Build the knowledge base

Run once to populate the Qdrant vector database with counseling conversations:

```bash
uv run python -m src.ingest_data
```

### API

**POST `/chat`**

```json
// Request
{ "message": "I've been feeling really anxious lately", "session_id": "abc-123" }

// Response
{
  "response": "I understand you're feeling anxious...",
  "intent": "asking_mental_health_question",
  "emotion": "fear",
  "language": "en"
}
```

**GET `/health`** — Returns service readiness status.

## Intent Categories

| Intent | Description |
|--------|-------------|
| `greeting` | Hi, hello, good morning, etc. |
| `goodbye` | Bye, take care, talk later, etc. |
| `gratitude` | Thanks, appreciate it, etc. |
| `asking_mental_health_question` | Personal distress, coping questions, mental health concerns |
| `out_of_scope` | Coding, trivia, medical advice, etc. |

## Multi-Language Support

The chatbot supports 20 languages with native keyword matching and LLM-based translation:

Arabic, Bulgarian, Chinese, Dutch, English, French, German, Greek, Hindi, Italian, Japanese, Polish, Portuguese, Russian, Spanish, Swahili, Thai, Turkish, Urdu, Vietnamese.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `language-identifier.ipynb` | Language detection model training |
| `model2-emotion.ipynb` | DistilBERT emotion classifier fine-tuning |

---

**Built with ❤️ for mental health support**
