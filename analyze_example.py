import joblib
import re
import numpy as np

model = joblib.load('spam_model.pkl')
vec = joblib.load('vectorizer.pkl')

example = "Earn $1000 daily from home. Limited offer!"
clean = re.sub('[^\\w\\s]', '', example.lower())
X = vec.transform([clean])
pred = model.predict(X)[0]

print('Example:', example)
print('Cleaned:', clean)
print('Prediction:', pred)
print('Model classes:', model.classes_)
try:
    df = model.decision_function(X)
    print('Decision function output:', df)
except Exception:
    print('Model has no decision_function output.')

# If linear model, show feature contributions
if hasattr(model, 'coef_'):
    coef_raw = model.coef_
    feat_names = vec.get_feature_names_out()
    x_arr = X.toarray()[0]
    # normalize coef to a flat numpy array when possible
    coef = None
    print('Raw coef type:', type(coef_raw))
    if hasattr(coef_raw, 'shape'):
        try:
            # sparse matrix or ndarray
            coef = np.array(coef_raw.toarray()).ravel()
        except Exception:
            try:
                coef = np.asarray(coef_raw).ravel()
            except Exception:
                coef = None
    else:
        # maybe an array-like (e.g., list) with one element containing the real coef
        try:
            possible = coef_raw[0]
            if hasattr(possible, 'toarray'):
                coef = np.array(possible.toarray()).ravel()
            else:
                coef = np.asarray(possible).ravel()
        except Exception:
            coef = None
    if coef is None:
        coef = np.zeros(len(feat_names), dtype=float)
    print('Feature count (vectorizer):', len(feat_names))
    print('Coef length (model):', coef.shape[0])

    nz_idx = np.where(x_arr > 0)[0]
    contributions = []
    for i in nz_idx:
        coef_i = float(coef[i]) if i < coef.shape[0] else 0.0
        contributions.append((feat_names[i], int(x_arr[i]), coef_i, coef_i * int(x_arr[i])))
    contributions.sort(key=lambda t: t[3], reverse=True)
    print('\nToken contributions (token, count, coef, contribution):')
    for t in contributions:
        print(t)
    # compute score using available coefficient entries
    score = 0.0
    limit = min(len(feat_names), coef.shape[0])
    score = float(np.dot(x_arr[:limit], coef[:limit]))
    print('\nRaw decision score (positive -> spam):', float(score))
    # include intercept to match decision_function
    try:
        intercept = float(model.intercept_[0])
        adjusted = float(score + intercept)
        print('Model intercept:', intercept)
        print('Adjusted score (score + intercept):', adjusted)
    except Exception:
        print('No intercept available')
    # show top overall spam/ham tokens where coef exists
    if coef.shape[0] >= 1:
        top_spam_idx = np.argsort(coef)[: -11 : -1]
        top_ham_idx = np.argsort(coef)[:10]
        print('\nTop spam tokens:', [(feat_names[i], float(coef[i])) for i in top_spam_idx if i < len(feat_names)])
        print('\nTop ham tokens:', [(feat_names[i], float(coef[i])) for i in top_ham_idx if i < len(feat_names)])
else:
    print('Model has no coef_ attribute; cannot show contributions.')
