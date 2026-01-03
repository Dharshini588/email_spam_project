import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import joblib

# Load and prepare data
data = pd.read_csv('data/spam.csv', encoding='latin-1')
spam = data[data['v1'] == 'spam']
ham = data[data['v1'] == 'ham']
min_n = min(len(spam), len(ham))
spam = spam.sample(min_n, random_state=42)
ham = ham.sample(min_n, random_state=42)
bal = pd.concat([spam, ham]).sample(frac=1, random_state=42).reset_index(drop=True)

x = bal['v2'].astype(str).str.lower().str.replace('[^\\w\\s]','', regex=True)
y = bal['v1'].astype(str)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

vec = CountVectorizer()
x_train_vec = vec.fit_transform(x_train)

model = SVC(kernel='linear')
model.fit(x_train_vec, y_train)

# Save
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vec, 'vectorizer.pkl')
print('Retrained and saved model and vectorizer.')
