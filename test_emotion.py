from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline
)

model = DistilBertForSequenceClassification.from_pretrained(
    "./models/distilbert-emotion-final"
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

print(classifier("I am very sad today"))