import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import re

data = pd.read_csv("data/spam.csv", encoding="latin-1")
# Use v1 (label) and v2 (text)
x = data['v2'].astype(str)
y = data['v1'].astype(str)

# Balance classes
spam = data[data['v1'] == 'spam']
ham = data[data['v1'] == 'ham']
min_n = min(len(spam), len(ham))
spam = spam.sample(min_n, random_state=42)
ham = ham.sample(min_n, random_state=42)
bal = pd.concat([spam, ham]).sample(frac=1, random_state=42).reset_index(drop=True)

x = bal['v2'].astype(str)
y = bal['v1'].astype(str)

# basic cleaning
x = x.str.lower().str.replace('[^\\w\\s]','', regex=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

vec = CountVectorizer()
x_train_vec = vec.fit_transform(x_train)
x_test_vec = vec.transform(x_test)

model = SVC(kernel='linear')
model.fit(x_train_vec, y_train)

pred = model.predict(x_test_vec)

print('Classification report:')
print(classification_report(y_test, pred))
print('Confusion matrix:')
print(confusion_matrix(y_test, pred))

# Show top features for spam vs ham (linear SVC supports coef_)
try:
    import numpy as np
    coef = model.coef_.toarray()[0]
    feature_names = vec.get_feature_names_out()
    top_spam = [(feature_names[i], coef[i]) for i in np.argsort(coef)[-20:]]
    top_ham = [(feature_names[i], coef[i]) for i in np.argsort(coef)[:20]]
    print('\nTop spam-associated tokens:')
    print(top_spam[::-1])
    print('\nTop ham-associated tokens:')
    print(top_ham)
except Exception as e:
    print('Could not extract coefficients:', e)

# Test some sample inputs
samples = [
    'Congratulations! You have won a free ticket. Call now to claim',
    'Hey, are we still meeting for lunch tomorrow?',
    'You have been selected for a prize; reply CLAIM to win',
    'See you at the meeting, please bring the report'
]
print('\nSample predictions:')
for s in samples:
    s_clean = re.sub('[^\\w\\s]','', s.lower())
    vec_s = vec.transform([s_clean])
    print(s, '->', model.predict(vec_s)[0])
