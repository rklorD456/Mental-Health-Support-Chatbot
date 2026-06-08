# 🧠 Mental Health Support Chatbot

A **Retrieval-Augmented Generation (RAG)** powered conversational AI system designed to provide empathetic mental health support. This project combines advanced NLP techniques including intent classification, emotion detection, language identification, and semantic retrieval to create a compassionate and context-aware chatbot.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Web Application](#running-the-web-application)
  - [Testing Intent Classifier](#testing-intent-classifier)
  - [Building the Knowledge Base](#building-the-knowledge-base)
- [Components](#components)
  - [Intent Classification](#intent-classification)
  - [Emotion Detection](#emotion-detection)
  - [Language Detection](#language-detection)
  - [RAG Pipeline](#rag-pipeline)
- [Data Ingestion](#data-ingestion)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Notebooks](#notebooks)
- [Development](#development)

## 📖 Overview

This project implements a **mental health support chatbot** that leverages:

1. **Intent Classification**: Routes user messages to appropriate handlers (greetings, mental health questions, etc.)
2. **Emotion Detection**: Identifies emotional states to tailor responses appropriately
3. **Language Detection**: Supports multi-language input detection
4. **RAG Pipeline**: Retrieves relevant mental health conversations from a vector database to provide informed, context-aware responses

The system is built with **FastAPI** for the backend and includes a web UI powered by Jinja2 templates. It uses the **Lightning AI API** (OpenAI-compatible, `openai/gpt-4o`) for LLM inference and **Qdrant** vector database for semantic search over mental health conversation data.

## 🏗️ Architecture

```
User Input
    ↓
┌─────────────────────────────────────┐
│   Intent Classification (Groq)      │
│   ├─ Greeting                       │
│   ├─ Goodbye                        │
│   ├─ Gratitude                      │
│   ├─ Mental Health Question         │
│   └─ Out of Scope                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Parallel Analysis                 │
│   ├─ Emotion Detection              │
│   │  (DistilBERT Fine-tuned)        │
│   └─ Language Detection             │
│      (scikit-learn LabelBinarizer)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   RAG Retrieval (Intent: Mental Q)  │
│   ├─ Embed query (BAAI/bge-base)    │
│   ├─ Search Qdrant                  │
│   └─ Get context with metadata      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Response Generation (Groq LLM)    │
│   ├─ Emotion-aware prompting        │
│   ├─ Context-enriched response      │
│   └─ Compassionate tone             │
└─────────────────────────────────────┘
    ↓
User Response + Metadata
```

## ✨ Features

- **Smart Intent Routing**: Classifies user messages into 5 distinct intent categories using Groq's language model
- **Emotion-Aware Responses**: Detects emotional states (sadness, fear, anger, joy, surprise, love) and adjusts response tone accordingly
- **Multi-Language Support**: Detects input language and can translate for processing
- **Semantic Search**: Retrieves relevant mental health conversations using vector embeddings
- **Web Interface**: User-friendly chat interface built with HTML/Jinja2
- **REST API**: Complete REST API with health checks and chat endpoints
- **FastAPI Auto-Documentation**: Interactive Swagger UI and ReDoc documentation
- **GPU-Optimized**: Automatically uses CUDA if available, falls back to CPU
- **Error Handling**: Graceful fallbacks for all critical operations

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | FastAPI 0.136+ |
| **LLM Integration** | Lightning AI API + OpenAI-compatible client |
| **Intent Classification** | OpenAI GPT-4o via Lightning AI |
| **Emotion Detection** | DistilBERT (fine-tuned custom model) |
| **Language Detection** | scikit-learn LabelBinarizer |
| **Embeddings** | Sentence Transformers (BAAI/bge-base-en-v1.5) |
| **Vector Database** | Qdrant Cloud |
| **Deep Learning** | PyTorch 2.10, Transformers 5.0 |
| **Data Loading** | HuggingFace Datasets |
| **Package Management** | UV (python-uv) |
| **Python Version** | 3.10 - 3.12 |

## 📁 Project Structure

```
Mental-Health-Support-Chatbot/
├── README.md                          # This file
├── pyproject.toml                     # Project configuration & dependencies
├── test_emotion.py                    # Test script for emotion detection
│
├── app/                               # Main application package
│   ├── main.py                        # FastAPI app with endpoints & model loading
│   ├── config.py                      # Configuration & settings management
│   ├── schemas.py                     # Pydantic models for requests/responses
│   ├── intent_classifier.py           # Intent classification logic
│   ├── rag_retriever.py              # RAG pipeline implementation
│   ├── router.py                      # Chat routing, emotion/language detection & translation
│   └── ingest_data.py                 # Data ingestion script for Qdrant
│
├── models/                            # Pre-trained models
│   ├── language_detector_v1.1.joblib  # Language detection model
│   └── distilbert-emotion-final/      # Fine-tuned emotion classifier
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── templates/                         # Web UI templates
│   └── index.html                     # Chat interface
│
└── Notebooks/                         # Jupyter notebooks for exploration
    ├── language-identifier.ipynb      # Language detection model development
    └── model2-emotion.ipynb           # Emotion classification model fine-tuning
```

## 🚀 Setup & Installation

### Prerequisites

- **Python**: 3.10 or later
- **Git**: For cloning the repository
- **UV Package Manager**: (Optional but recommended)
  ```bash
  pip install uv
  ```

### Environment Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone https://github.com/rklorD456/Mental-Health-Support-Chatbot.git
   cd Mental-Health-Support-Chatbot
   ```

2. **Create environment file** (`.env`):
   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env   # Linux/macOS
   ```

3. **Configure environment variables** in `.env`:
   ```env
   # Lightning AI API (used for intent classification, RAG, and translation)
   LIGHTNING_API_KEY=your_lightning_api_key_here

   # Qdrant Vector Database Configuration
   QDRANT_URL=https://your-qdrant-cloud-url.qdrant.io
   QDRANT_API_KEY=your_qdrant_api_key_here
   ```

4. **Install dependencies**:
   ```bash
   # Using UV (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

Configuration is centralized in [config.py](app/config.py):

```python
class Settings:
    # API credentials
    lightning_api_key          # Lightning AI API key
    lightning_base_url         # Lightning AI endpoint
    qdrant_url                # Qdrant cloud URL
    qdrant_api_key            # Qdrant API key

    # Model configuration
    model_used_api            # Active LLM API key (points to lightning_api_key)
    model_used_base_url       # Active LLM base URL (points to lightning_base_url)
    model_used_name           # "openai/gpt-4o"
    embedding_model_name      # "BAAI/bge-base-en-v1.5"
    collection_name           # "mental_health_knowledge_base"
    temperature               # 0.3

    # Paths
    abs_language_model_path   # Path to language detector model
    abs_emotion_model_path    # Path to emotion classifier model
    templates_dir             # Templates directory path
```

## 💬 Usage

### Running the Web Application

Start the FastAPI server with auto-reload for development:

```bash
uv run uvicorn app.main:app --reload
```

The application will be available at `http://localhost:8000`

**Features:**
- Web chat interface at the root URL
- Interactive API documentation at `/docs` (Swagger UI)
- Alternative API docs at `/redoc` (ReDoc)
- Health check endpoint at `/health`

### Testing Intent Classifier

Test the intent classification module from the command line:

```bash
# Test with a custom message
uv run python app/intent_classifier.py "I have been feeling really overwhelmed and stressed out lately."

# Test with the default message
uv run python app/intent_classifier.py
```

Expected output:
```json
{
  "intent": "asking_mental_health_question"
}
```

### Building the Knowledge Base

The [ingest_data.py](app/ingest_data.py) script populates the Qdrant vector database with mental health conversation data:

```bash
uv run python app/ingest_data.py
```

**What it does:**
1. Connects to Qdrant Cloud
2. Creates the `mental_health_knowledge_base` collection if it doesn't exist
3. Downloads the `Amod/mental_health_counseling_conversations` dataset from HuggingFace
4. Encodes all conversations using `BAAI/bge-base-en-v1.5` embeddings
5. Stores vectors with metadata (question, answer) in Qdrant
6. Creates 768-dimensional vector embeddings with cosine distance metric

**Dataset Details:**
- **Source**: Hugging Face [Amod/mental_health_counseling_conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
- **Size**: Thousands of real counseling conversations
- **Format**: Patient context + Counselor responses
- **Embedding Model**: BAAI/bge-base-en-v1.5 (768-dim, optimized for semantic search)

## 🧩 Components

### Intent Classification

**File**: [intent_classifier.py](app/intent_classifier.py)

Classifies user messages into one of five intent categories using Groq's language model:

| Intent | Description |
|--------|-------------|
| `greeting` | Casual greetings, introductions |
| `goodbye` | Farewell messages, session endings |
| `gratitude` | Thank you messages, appreciation |
| `asking_mental_health_question` | Mental health concerns, emotional support requests |
| `out_of_scope` | Questions outside mental health domain |

**Key Features:**
- Uses `openai/gpt-4o` model via Lightning AI API
- Zero temperature (deterministic output)
- Strict JSON output format validation
- Priority handling: Mental health questions prioritized over other intents
- Graceful error handling with fallback to "out_of_scope"

**Example:**
```python
from app.intent_classifier import get_intent

result = get_intent("I'm feeling anxious about my upcoming presentation")
print(result)  # {"intent": "asking_mental_health_question"}
```

### Emotion Detection

**File**: [router.py](app/router.py) → `detect_emotion()`

Identifies emotional states in user messages using a fine-tuned DistilBERT model:

**Supported Emotions:**
- `sadness` → Compassionate, gentle tone
- `fear` → Calm, reassuring tone
- `anger` → Non-escalating, validating tone
- `joy` → Warm, encouraging tone
- `surprise` → Steady, grounding tone
- `love` → Supportive, connected tone

**Technical Details:**
- **Base Model**: DistilBERT (smaller, faster than BERT)
- **Fine-tuning**: Custom trained on mental health datasets
- **Device**: Auto-detects GPU (CUDA) or falls back to CPU
- **Truncation**: Handles long texts (max 512 tokens)

**Example:**
```python
from app.router import detect_emotion

emotion = detect_emotion("I'm so angry about what happened!")
print(emotion)  # "anger"
```

### Language Detection

**File**: [router.py](app/router.py) → `detect_language()`

Identifies the language of input text:

**Technical Details:**
- **Model**: scikit-learn trained LabelBinarizer classifier
- **Format**: Saved as joblib file (`language_detector_v1.1.joblib`)
- **Output**: Language code (e.g., "en", "es", "fr")

**Example:**
```python
from app.router import detect_language

lang = detect_language("Hello, how are you today?")
print(lang)  # "en"
```

### RAG Pipeline

**File**: [rag_retriever.py](app/rag_retriever.py)

Implements Retrieval-Augmented Generation for context-aware responses:

**Workflow:**
1. **Query Embedding**: Convert user message to 768-dim vector using BAAI/bge-base-en-v1.5
2. **Vector Search**: Query Qdrant for semantically similar conversations
3. **Context Retrieval**: Extract top results with patient context and counselor responses
4. **Prompt Augmentation**: Inject retrieved context into system prompt
5. **Response Generation**: Use Groq LLM with emotion-aware instructions

**Key Features:**
- Emotion-aware tone mapping (sadness → gentle, anger → validating, etc.)
- Uses `openai/gpt-4o` model for response generation
- Temperature 0.3 for balanced creativity and consistency
- Includes metadata from retrieved conversations for better context

**Example:**
```python
from app.rag_retriever import get_rag_response

response = get_rag_response(
    user_message="I'm struggling with anxiety",
    emotion="fear"
)
print(response)  # Generated response with retrieved context
```

## 📊 Data Ingestion

The knowledge base is built using real mental health counseling conversations:

### Dataset Source
- **HuggingFace**: `Amod/mental_health_counseling_conversations`
- **Contains**: Patient context + Professional counselor responses
- **Purpose**: Provides grounded examples for RAG retrieval

### Ingestion Process

```python
# app/ingest_data.py workflow:

1. Connect to Qdrant Cloud
2. Check if collection exists, create if needed
3. Load HuggingFace dataset
4. For each conversation:
   - Combine patient context + counselor response
   - Generate embedding (BAAI/bge-base-en-v1.5)
   - Store as vector with metadata in batches
5. Complete indexing and optimize collection
```

### Vector Store Details
- **Collection**: `mental_health_knowledge_base`
- **Vector Dimension**: 768 (from BAAI/bge model)
- **Distance Metric**: Cosine similarity
- **Metadata Stored**: Patient context, counselor response
- **Batch Size**: 100 vectors per batch (for efficiency)

## 🤖 Model Details

### Language Detection Model
- **File**: `models/language_detector_v1.1.joblib`
- **Type**: scikit-learn LabelBinarizer classifier
- **Loaded**: At application startup via JobLib
- **Output**: Language code

### Emotion Classification Model
- **Directory**: `models/distilbert-emotion-final/`
- **Architecture**: DistilBERT (transformer-based)
- **Fine-tuned**: On mental health emotion datasets
- **Files**:
  - `config.json` - Model architecture configuration
  - `model.safetensors` - Weights (efficient, secure format)
  - `tokenizer.json` - BPE tokenizer
  - `tokenizer_config.json` - Tokenizer configuration
- **Inference**: Loaded via HuggingFace `pipeline()` at startup
- **Device**: Auto-selects GPU if CUDA available

## 🔌 API Endpoints

### REST Endpoints

**Base URL**: `http://localhost:8000`

#### GET `/`
Returns the web chat interface
```bash
curl http://localhost:8000/
```

#### GET `/health`
Health check endpoint
```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "ok",
  "models_loaded": ["language", "emotion"]
}
```

#### POST `/chat`
Main chat endpoint for getting responses
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am feeling really anxious lately"}'
```

**Request Schema**:
```json
{
  "message": "string (required, non-empty)"
}
```

**Response Schema**:
```json
{
  "response": "string (generated response)",
  "intent": "string (one of: greeting, goodbye, gratitude, asking_mental_health_question, out_of_scope)",
  "emotion": "string (detected emotion: sadness, fear, anger, joy, surprise, love, neutral)",
  "language": "string (detected language code)"
}
```

**Example Response**:
```json
{
  "response": "I understand you're feeling anxious. That's a completely valid emotion...",
  "intent": "asking_mental_health_question",
  "emotion": "fear",
  "language": "en"
}
```

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

Both provide interactive API testing and detailed endpoint documentation.

## 📓 Notebooks

Two Jupyter notebooks document the model development process:

### 1. [language-identifier.ipynb](Notebooks/language-identifier.ipynb)

Covers language detection model development:
- Data exploration and preprocessing
- Feature engineering for language identification
- Model training and evaluation
- Performance metrics and confusion matrices
- Model persistence and serialization

### 2. [model2-emotion.ipynb](Notebooks/model2-emotion.ipynb)

Details emotion classification model fine-tuning:
- Dataset loading and exploration
- Baseline model evaluation
- DistilBERT fine-tuning workflow
- Hyperparameter optimization
- Emotion distribution analysis
- Training curves and loss visualization
- Model evaluation metrics and per-emotion performance
- Production model export and testing

## 🔧 Development

### Project Dependencies

See [pyproject.toml](pyproject.toml) for the complete dependency list:

**Core Dependencies:**
- `fastapi>=0.136.3` - Web framework
- `groq>=1.4.0` - Groq API client
- `openai>=2.41.0` - OpenAI-compatible client
- `transformers==5.0.0` - HuggingFace transformer models
- `torch==2.10.0` - Deep learning framework
- `sentence-transformers>=5.5.1` - Embedding models
- `qdrant-client>=1.18.0` - Vector database client
- `datasets` - HuggingFace datasets library
- `joblib>=1.5.3` - Model persistence
- `python-dotenv>=0.9.9` - Environment variable management
- `jinja2>=3.1.6` - Template engine
- `streamlit>=1.58.0` - Optional UI framework

### Performance Tips

1. **GPU Acceleration**: Ensure PyTorch can access GPU for faster emotion detection
2. **Batch Processing**: The ingestion script batches vectors for efficiency
3. **Connection Pooling**: Qdrant client manages connection pooling automatically
4. **Model Caching**: Models are loaded once at startup, reused for all requests

### Error Handling

The system includes graceful fallbacks:
- Language detection fails → defaults to "en"
- Emotion detection fails → defaults to "neutral"
- Intent classification fails → defaults to "out_of_scope"
- RAG retrieval fails → falls back to direct response generation

## 📝 Notes & Future Enhancements

- **Current State**: Fully functional RAG-based chatbot with multi-modal NLP analysis
- **Data Privacy**: When deploying, ensure compliance with mental health data regulations
- **Crisis Handling**: The intent classifier explicitly avoids providing crisis responses; route emergencies to appropriate resources
- **Model Updates**: Both emotion and language models can be retrained with new data using the notebooks
- **Multi-language Support**: Full translation pipeline is implemented — non-English input is translated to English for processing, and responses are translated back to the user's language
- **Monitoring**: Integrate with observability tools for production monitoring

---

**Built with ❤️ for mental health support**
