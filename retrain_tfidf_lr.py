import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv('data/spam.csv', encoding='latin-1')
# Keep columns v1 (label) and v2 (text)
spam = data[data['v1'] == 'spam']
ham = data[data['v1'] == 'ham']
min_n = min(len(spam), len(ham))
spam = spam.sample(min_n, random_state=42)
ham = ham.sample(min_n, random_state=42)
bal = pd.concat([spam, ham]).sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocess: lowercase and remove punctuation (keep numbers/words/whitespace)
X = bal['v2'].astype(str).str.lower().str.replace('[^\\w\\s]', '', regex=True)
y = bal['v1'].astype(str)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF with unigrams + bigrams
vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# Logistic Regression (balanced classes)
model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
print('Training LogisticRegression on TF-IDF features...')
model.fit(X_train_vec, y_train)

# Evaluate
pred = model.predict(X_test_vec)
print('\nClassification report:')
print(classification_report(y_test, pred))
print('Confusion matrix:')
print(confusion_matrix(y_test, pred))

# Save model and vectorizer (overwrite previous files so demo loads the improved model)
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vec, 'vectorizer.pkl')
print('\nSaved spam_model.pkl and vectorizer.pkl')
