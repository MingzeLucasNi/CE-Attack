import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# Define the paths to your datasets
splits = {
    'train': 'en-zh/train-00000-of-00001.parquet',
    'test': 'en-zh/test-00000-of-00001.parquet',
    'validation': 'en-zh/validation-00000-of-00001.parquet'
}

# df_test = pd.read_parquet("hf://datasets/wmt/wmt20_mlqe_task1/" + splits["test"])['translation']

splits = {'train': 'en-de/train-00000-of-00001.parquet', 'test': 'en-de/test-00000-of-00001.parquet', 'validation': 'en-de/validation-00000-of-00001.parquet'}
# df = pd.read_parquet("hf://datasets/wmt/wmt20_mlqe_task2/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/wmt/wmt20_mlqe_task2/" + splits["test"])['translation']

# Load the test set using pandas

# Choose a pre-trained model and tokenizer
model_name = "gpt2"  # You can choose other models as well
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Set pad_token to eos_token to handle padding
tokenizer.pad_token = tokenizer.eos_token

# Function to tokenize the dataset
def tokenize_text(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Function to compute perplexity for a single example
def compute_perplexity(inputs):
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

# Tokenize the entire test set
tokenized_texts = [tokenize_text(text['en']) for text in df_test]

# Calculate perplexity for each example in the test set
perplexities = []
for tokenized_text in tokenized_texts:
    # Move the inputs to the model's device (e.g., GPU if available)
    tokenized_text = {k: v.to(model.device) for k, v in tokenized_text.items()}
    perplexity = compute_perplexity(tokenized_text)
    perplexities.append(perplexity)

# Compute the average perplexity
average_perplexity = np.mean(perplexities)
print(f"Average Perplexity: {average_perplexity}")
