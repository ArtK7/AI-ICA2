import torch
from transformers import AutoModelForSequenceClassification


def load_model(model_name, device):
    print(f"Loading model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    return model


def evaluate_model(model, tokenizer, balanced_data, device):
    inputs = tokenizer(
        [ex["text"] for ex in balanced_data],
        padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    true_labels = torch.tensor([ex["label"] for ex in balanced_data]).to(device)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return true_labels, predictions