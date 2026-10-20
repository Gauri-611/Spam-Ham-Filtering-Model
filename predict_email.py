import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "spam_ham_model.joblib"
VECTORIZER_PATH = ROOT / "tfidf_vectorizer.joblib"


def ensure_model_exists():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        from train_model import train_model

        train_model()


def predict_email(email: str):
    ensure_model_exists()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    prediction = model.predict(vectorizer.transform([email]))[0]
    return prediction.upper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an email is spam or ham.")
    parser.add_argument("message", nargs="?", default="", help="The email text to classify.")
    args = parser.parse_args()

    email_text = args.message.strip()

    if not email_text:
        email_text = input("Enter an email message: ").strip()

    prediction = predict_email(email_text)
    print(f"Prediction: {prediction}")
    if prediction == "SPAM":
        print("This is a spam email.")
    else:
        print("This is not a spam email.")
