# Spam Ham Filtering Model

A simple machine learning project that classifies emails as `spam` or `ham` using a TF-IDF + Multinomial Naive Bayes pipeline.

## What this project does

- Loads a small email dataset
- Splits the data into training and testing sets
- Converts email text into numerical vectors with TF-IDF
- Trains a Naive Bayes classifier
- Saves the trained model and vectorizer
- Predicts whether a new email is spam or ham

## Project structure

- `data/spam_ham_dataset.csv` - email dataset used for training
- `train_model.py` - trains the model and saves artifacts
- `predict_email.py` - loads the trained model and predicts a new email
- `requirements.txt` - Python dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This will train the model and save:

- `spam_ham_model.joblib`
- `tfidf_vectorizer.joblib`

## Predict a new email

```bash
python predict_email.py "Congratulations! You have won a free prize"
```

Example output:

```text
Prediction: SPAM
```

## Notes

This is a beginner-friendly demo project, not a production-grade spam filter. For real-world use, you would usually want a larger dataset, better preprocessing, cross-validation, and more robust evaluation.
