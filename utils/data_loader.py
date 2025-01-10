from datasets import load_dataset
from transformers import AutoTokenizer


def load_imdb_dataset():
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    return dataset


def tokenize_dataset(dataset, tokenizer_name):
    print("Tokenizing the dataset...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, desc="Tokenizing Data")
    return tokenized_datasets