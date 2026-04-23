"""
if you want to test the model and see the result you can run this file 
you have to provide your sentences in "input.txt" file (one sentence per line) and it will classify each sentence as normal or hate/offensive
the output will be saved in "output.txt" file in the format:
sentence --> label
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =========================
# 1. LOAD MODEL + TOKENIZER
# =========================
path = "./sentiment_classifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

model.to(device)
model.eval()

# label mapping (same as training)
labels = {0: "normal", 1: "hate/offensive"}

# =========================
# 2. FILE PATHS
# =========================
input_file = "input.txt"
output_file = "output.txt"

# =========================
# 3. PROCESS FILE
# =========================
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

results = []

for line in lines:
    text = line.strip()

    if not text:
        continue

    # tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # prediction
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    label = labels[pred]

    # append result to same sentence
    results.append(f"{text} --> {label}")

# =========================
# 4. SAVE OUTPUT FILE
# =========================
with open(output_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(r + "\n")

print(f"Done! Results saved to: {output_file}")