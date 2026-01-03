import joblib
import re

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("✅ Spam Detection Model Loaded Successfully!\n")

examples = [
    "Congratulations! You won a free iPhone. Click here now!",
    "Hey, can we meet tomorrow at 10am for the project?",
    "Earn $1000 daily from home. Limited offer!",
    "Please find attached the homework for review."
]

for i, email in enumerate(examples, start=1):
    # Preprocess the email the same way as training (lowercase + remove punctuation)
    cleaned = re.sub('[^\\w\\s]', '', email.lower())
    email_vector = vectorizer.transform([cleaned])
    prediction = model.predict(email_vector)

    # show probabilities if available
    probs = None
    try:
        probs = model.predict_proba(email_vector)[0]
    except Exception:
        pass

    if probs is not None:
        prob_map = dict(zip(model.classes_, probs))
        probs_str = ' '.join([f"{c}: {prob_map.get(c,0):.3f}" for c in model.classes_])
        print(f"Example {i}: {prediction[0]} → {email} ({probs_str})")
    else:
        print(f"Example {i}: {prediction[0]} → {email}")
