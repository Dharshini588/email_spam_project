import argparse
import time
import pandas as pd
import joblib
import re

data = pd.read_csv("data/spam.csv", encoding="latin-1")
x_data = data["v2"]
y_data = data["v1"]


def print_dataset_preview(x_series, y_series, n=5):
    print("\n" + "=" * 60)
    print(f"Dataset preview (first {n} emails)")
    print("-" * 60)
    for i, txt in enumerate(x_series.head(n), start=1):
        # truncate long emails for preview
        snippet = txt.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        print(f"{i}. {snippet}")
    print("-" * 60)
    print("Label counts:")
    counts = y_series.value_counts()
    for lbl, cnt in counts.items():
        print(f"  {lbl:5}: {cnt}")
    print("=" * 60 + "\n")

print_dataset_preview(x_data, y_data)

from pathlib import Path

# Load the retrained model and vectorizer (TF-IDF + LogisticRegression)
model_path = Path("spam_model.pkl")
vec_path = Path("vectorizer.pkl")
if model_path.exists() and vec_path.exists():
    model = joblib.load(str(model_path))
    vectorizer = joblib.load(str(vec_path))
    print(f"Loaded saved model: {model_path} and vectorizer: {vec_path}")
else:
    raise FileNotFoundError("Saved model or vectorizer not found. Run retrain_tfidf_lr.py first.")

parser = argparse.ArgumentParser(description="Spam/ham prediction demo")
parser.add_argument("--message", "-m", help="Provide an email message on the command line (non-interactive)")
args = parser.parse_args()

print("\n--- Test with your own email ---")
if args.message:
    user_email = args.message
    print("Using message from --message flag (non-interactive).")
else:
    user_email = input("Enter an email message: ")

# Preprocess the user input the same way as training data
start = time.time()
clean_email = user_email.lower()
clean_email = re.sub('[^\\w\\s]', '', clean_email)

# Convert text to numbers using the SAME vectorizer
user_email_vector = vectorizer.transform([clean_email])

# Predict spam or ham and show probabilities if available
pred = model.predict(user_email_vector)[0]
print("\n" + "=" * 60)
print(f"Prediction: {pred.upper()}")
if hasattr(model, "predict_proba"):
    try:
        probs = model.predict_proba(user_email_vector)[0]
        # map probabilities to classes and sort
        class_prob = dict(zip(model.classes_, probs))
        sorted_probs = sorted(class_prob.items(), key=lambda kv: kv[1], reverse=True)
        print("Confidence:")
        for c, p in sorted_probs:
            print(f"  {c:5}: {p*100:6.2f}%")
    except Exception:
        print("(Model does not provide probability estimates.)")
else:
    print("(Model does not provide probability estimates.)")
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.3f}s")
print("=" * 60)






