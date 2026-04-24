from transformers import pipeline

# =========================
# 1. LOAD PRETRAINED MODEL
# =========================

classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-offensive"
)

# =========================
# 2. FILE PATHS
# =========================
input_file = "input.txt"
output_file = "output.txt"

# =========================
# 3. READ INPUT
# =========================
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

results = []

# =========================
# 4. CLASSIFY EACH LINE
# =========================
for line in lines:
    text = line.strip()
    print(f"Classifying: {text}")
    if not text:
        continue
    prediction = classifier(text)[0]
    label = prediction["label"]
    score = prediction["score"]

    results.append(f"{text} --> {label} ({score:.2f})")

# =========================
# 5. WRITE OUTPUT
# =========================
with open(output_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(r + "\n")

print("Done! Results saved to output.txt")