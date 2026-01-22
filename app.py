from flask import Flask, request, render_template
import joblib
import re
import os

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    batch_results = None
    rating = None
    contact_success = False
    text_value = ''
    if request.method == 'POST':
        # Handle contact form
        if 'contact_submit' in request.form:
            name = request.form.get('contact_name')
            email = request.form.get('contact_email')
            message = request.form.get('contact_message')
            # Here you could add logic to store or send the message
            contact_success = True
        # Handle rating form
        if 'rate' in request.form and 'rating' in request.form:
            rating = int(request.form['rating'])
        # Handle batch file upload
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file and file.filename.endswith('.txt'):
                lines = file.read().decode('utf-8', errors='ignore').splitlines()
                batch_results = []
                for line in lines:
                    clean_email = clean_text(line)
                    email_vector = vectorizer.transform([clean_email])
                    pred = model.predict(email_vector)[0]
                    conf = None
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(email_vector)[0]
                        conf = dict(zip(model.classes_, probs))
                    label = pred.upper()
                    if label == 'HAM':
                        label = 'NOT SPAM'
                    batch_results.append({'email': line, 'prediction': label, 'confidence': conf})
        elif 'email' in request.form:
            email = request.form['email']
            text_value = email
            clean_email = clean_text(email)
            email_vector = vectorizer.transform([clean_email])
            pred = model.predict(email_vector)[0]
            label = pred.upper()
            if label == 'HAM':
                label = 'NOT SPAM'
            prediction = label
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(email_vector)[0]
                confidence = dict(zip(model.classes_, probs))
    return render_template('index.html', prediction=prediction, confidence=confidence, batch_results=batch_results, rating=rating, contact_success=contact_success, text_value=text_value)

if __name__ == '__main__':
    app.run(debug=True)
