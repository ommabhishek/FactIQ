import pickle
from preprocess import clean_text

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_news(news_text, model, vectorizer):
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    confidence = model.predict_proba(vector).max() * 100
    label = "REAL NEWS" if prediction[0] == "True" else "FAKE NEWS"
    return label, round(confidence, 2)

if __name__ == "__main__":
    model, vectorizer = load_model()
    text = input("Enter news text: ")
    result, confidence = predict_news(text, model, vectorizer)
    print(f"Prediction: {result} (Confidence: {confidence}%)")
