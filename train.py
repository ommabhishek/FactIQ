import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text

def train():
    # Load data
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake['label'] = 'FAKE'
    true['label'] = 'True'

    combined = pd.concat([fake, true], axis=0)
    combined = combined.sample(frac=1).reset_index(drop=True)
    combined = combined[['title', 'text', 'label']]
    combined.to_csv("news_dataset.csv", index=False)

    # Preprocess
    combined['content'] = combined['title'] + " " + combined['text']
    combined['content'] = combined['content'].apply(clean_text)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(combined['content'])
    y = combined['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model and vectorizer saved!")

if __name__ == "__main__":
    train()
