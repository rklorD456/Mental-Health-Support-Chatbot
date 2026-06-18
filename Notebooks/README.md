# Notebooks/

Jupyter notebooks used to train and evaluate the custom NLP models.

## Files

| Notebook | Purpose |
|----------|---------|
| `language-identifier.ipynb` | Trains the scikit-learn language detection model. Outputs `models/language_detector_v1.1.joblib`. |
| `model2-emotion.ipynb` | Fine-tunes a DistilBERT model for emotion classification (sadness, fear, anger, joy, surprise, love). Outputs `models/distilbert-emotion-final/`. |

These notebooks are for **training only** — the trained models are loaded at runtime from the `models/` directory.
