import streamlit as st
import pickle
from src.preprocess import clean_text

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# UI
st.title("📰 Fake News Detection System")
st.markdown("Paste a news **headline or article** below to check if it's Real or Fake.")

st.divider()

news_input = st.text_area("Enter News Text:", height=200, placeholder="Paste your news article or headline here...")

if st.button("🔍 Analyze", use_container_width=True):
    if news_input.strip() == "":
        st.warning("Please enter some news text first!")
    else:
        cleaned = clean_text(news_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max() * 100

        st.divider()

        if prediction == "True":
            st.success(f"✅ REAL NEWS  —  Confidence: {confidence:.2f}%")
        else:
            st.error(f"🚨 FAKE NEWS  —  Confidence: {confidence:.2f}%")

        st.progress(int(confidence))
        st.caption(f"Model confidence: {confidence:.2f}%")

st.divider()
st.markdown("<center>Built with ❤️ for GSSoC | Fake News Detection System</center>", unsafe_allow_html=True)
