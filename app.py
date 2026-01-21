import streamlit as st
import joblib
from pathlib import Path

# 1. Load your saved model and vectorizer (use the actual filenames in this repo)
MODEL_PATH = Path("../spam_model.pkl").resolve() if Path("model.pkl").parent.name == 'main.py' else Path("spam_model.pkl")
VECT_PATH = Path("vectorizer.pkl")

try:
    model = joblib.load(str(MODEL_PATH))
    vectorizer = joblib.load(str(VECT_PATH))
except FileNotFoundError:
    st.error(f"Error: Could not find '{MODEL_PATH.name}' or '{VECT_PATH.name}'. Run retrain scripts first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# 2. Create the Title and Description
st.title("📧 Email Spam Detector")
st.write("Paste an email below to see if it's spam or safe.")

# 3. Create a Text Box for the user
user_input = st.text_area("Enter email text here:", height=150)

# 4. Create a Button to click
if st.button("Check Email"):
    if user_input:
        # 5. Transform the text (just like you did in your notebook)
        data = [user_input]
        vectorized_data = vectorizer.transform(data)

        # 6. Make the prediction
        try:
            result = model.predict(vectorized_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            raise

        # 7. Show the result robustly (support string labels like 'spam'/'ham')
        label = result[0]
        if isinstance(label, bytes):
            try:
                label = label.decode()
            except Exception:
                pass

        if str(label).lower() in ("spam", "1", "true", "t"):
            st.header("🚨 This is SPAM!")
        else:
            st.header("✅ This is Safe (Ham).")
        # show probabilities if available
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(vectorized_data)[0]
                prob_map = dict(zip(model.classes_, probs))
                st.write("**Confidence:**")
                for lbl, p in sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True):
                    st.write(f"- {lbl}: {p*100:5.2f}%")
            except Exception:
                st.info("No probability estimates available for this model.")
    else:
        st.warning("Please enter some text first.")