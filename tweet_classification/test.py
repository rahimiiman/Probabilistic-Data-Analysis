"""
if you want to test the model and see the result you can run this file 
it will ask you to enter a sentence and it will classify it as normal or hate/offensive
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
# 2. INTERACTIVE LOOP
# =========================
print("\n--- Hate Speech Detector ---")
print("Type a sentence to classify")
print("Type 0 to exit\n")

while True:
    text = input("Enter text: ")

    if text == "0":
        print("Exiting...")
        break

    # tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # prediction
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    print("Prediction:", labels[pred])
    print("-" * 40)