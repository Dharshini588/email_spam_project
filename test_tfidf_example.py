import joblib
import re

model = joblib.load('spam_model.pkl')
vec = joblib.load('vectorizer.pkl')

examples = [
    'Earn $1000 daily from home. Limited offer!',
    'Congratulations! You have won a free ticket. Call now to claim',
    'Hey, are we still meeting for lunch tomorrow?'
]

for e in examples:
    cleaned = re.sub('[^\\w\\s]', '', e.lower())
    X = vec.transform([cleaned])
    pred = model.predict(X)[0]
    probs = None
    try:
        probs = model.predict_proba(X)[0]
    except Exception:
        pass
    print('\nExample:', e)
    print('Cleaned:', cleaned)
    print('Prediction:', pred)
    if probs is not None:
        # map to classes
        for cls, p in zip(model.classes_, probs):
            print(f'  prob({cls}) = {p:.4f}')
