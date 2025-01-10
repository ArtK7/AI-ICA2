import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer
from utils.data_loader import load_imdb_dataset, tokenize_dataset
from utils.model_utils import load_model, evaluate_model
import torch
from colorama import Fore, Style  # For colorful output

# =======================
# Configuration
# =======================
MODEL_NAMES = [
    "distilbert-base-uncased-finetuned-sst-2-english",  # Fine-tuned on SST-2
    "textattack/distilbert-base-uncased-imdb"           # Fine-tuned on IMDb
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Store results for both models to summarize later
model_results = []

# =======================
# Script Header
# =======================
print(Fore.MAGENTA + "\n" + "=" * 50)
print("DistilBERT Model Testing Script".center(50))
print("=" * 50 + Style.RESET_ALL)

# =======================
# Step 1: Load IMDb Dataset
# =======================
print(Fore.MAGENTA + "\nStep 1: Loading IMDb dataset..." + Style.RESET_ALL)
dataset = load_imdb_dataset()

# =======================
# Step 2: Tokenize the Dataset
# =======================
print(Fore.MAGENTA + "\nStep 2: Tokenizing the dataset..." + Style.RESET_ALL)
tokenized_datasets = tokenize_dataset(dataset, MODEL_NAMES[0])  # Tokenize once using the first model's tokenizer

# =======================
# Step 3: Balance the Dataset
# =======================
print(Fore.MAGENTA + "\nStep 3: Balancing the dataset..." + Style.RESET_ALL)
test_data = tokenized_datasets["test"].shuffle(seed=42)

# Separate positive and negative samples
positive_samples = [ex for ex in test_data if ex['label'] == 1]
negative_samples = [ex for ex in test_data if ex['label'] == 0]

# Create a balanced dataset with equal positive and negative samples
sample_size = 250 // 2  # 125 positive, 125 negative
balanced_data = random.sample(positive_samples, sample_size) + random.sample(negative_samples, sample_size)
random.shuffle(balanced_data)

print(Fore.GREEN + f"Balanced dataset created with {len(balanced_data)} samples (125 positive, 125 negative).\n" + Style.RESET_ALL)

# =======================
# Step 4: Test Both Models
# =======================
for model_name in MODEL_NAMES:
    print(Fore.MAGENTA + "=" * 50 + Style.RESET_ALL)
    print(Fore.MAGENTA + f"Step 4: Testing model: {model_name}".center(50) + Style.RESET_ALL)
    print(Fore.MAGENTA + "=" * 50 + Style.RESET_ALL)

    # Load the model and tokenizer
    print(Fore.WHITE + f"\nLoading model: {model_name}..." + Style.RESET_ALL)
    model = load_model(model_name, DEVICE)  # Load the model (fine-tuned or base)
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load the tokenizer

    # Evaluate the model on the balanced dataset
    print(Fore.WHITE + "Evaluating model performance..." + Style.RESET_ALL)
    true_labels, predictions = evaluate_model(model, tokenizer, balanced_data, DEVICE)

    # Ensure predictions is a tensor (use torch.tensor if needed)
    predictions = torch.tensor(predictions) if not isinstance(predictions, torch.Tensor) else predictions

    # Count positive and negative predictions
    positive_count = (predictions == 1).sum().item()  # Works correctly for tensors
    negative_count = (predictions == 0).sum().item()

    # Display positive and negative review counts in green and red
    print(Fore.GREEN + f"Positive Reviews: {positive_count}" + Style.RESET_ALL)
    print(Fore.RED + f"Negative Reviews: {negative_count}" + Style.RESET_ALL)

    # Compute performance metrics: accuracy, precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels.cpu().tolist(), predictions.cpu().tolist(), average='binary', zero_division=0
    )
    accuracy = accuracy_score(true_labels.cpu().tolist(), predictions.cpu().tolist())

    # Store results for summary
    model_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    # Display results for this model
    print(Fore.MAGENTA + f"\nResults for {model_name}:" + Style.RESET_ALL)
    print(Fore.WHITE + f"{'Metric':<12}{'Value':<10}" + Style.RESET_ALL)
    print(Fore.WHITE + f"{'-' * 22}" + Style.RESET_ALL)
    print(f"{Fore.MAGENTA}{'Accuracy':<12}{accuracy:.2f}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'Precision':<12}{precision:.2f}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'Recall':<12}{recall:.2f}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'F1 Score':<12}{f1:.2f}{Style.RESET_ALL}")

# =======================
# Final Summary
# =======================
print(Fore.MAGENTA + "\n" + "=" * 50)
print("Final Summary of Model Performance".center(50))
print("=" * 50 + Style.RESET_ALL)

for result in model_results:
    print(Fore.WHITE + f"Model: {result['Model']}" + Style.RESET_ALL)
    print(f"{Fore.MAGENTA}{'Accuracy:':<12}{result['Accuracy']:.2f}" + Style.RESET_ALL)
    print(f"{Fore.MAGENTA}{'Precision:':<12}{result['Precision']:.2f}" + Style.RESET_ALL)
    print(f"{Fore.MAGENTA}{'Recall:':<12}{result['Recall']:.2f}" + Style.RESET_ALL)
    print(f"{Fore.MAGENTA}{'F1 Score:':<12}{result['F1 Score']:.2f}" + Style.RESET_ALL)
    print(Fore.WHITE + "-" * 30 + Style.RESET_ALL)