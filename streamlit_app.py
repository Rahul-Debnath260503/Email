import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
with open('spam_voting_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("📧 Spam Email Classifier")
st.write("Enter a message below to detect if it's spam or not.")

message = st.text_area("Email content:")

if st.button("Predict"):
    if not message.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        cleaned = preprocess(message)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)

        if pred[0] == 1:
            st.error("🚫 This is **SPAM**!")
        else:
            st.success("✅ This is **NOT SPAM**.")
