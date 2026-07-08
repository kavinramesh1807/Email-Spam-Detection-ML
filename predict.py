# ==========================================================
# EMAIL SPAM DETECTION - PREDICTION PROGRAM
# ==========================================================

import os
import re
import string
import joblib

# Create paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_detection_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("=" * 60)
print("        EMAIL SPAM DETECTION SYSTEM")
print("=" * 60)

while True:
    message = input("\nEnter a message to check spam/ham: ")

    cleaned_message = clean_text(message)

    message_tfidf = tfidf.transform([cleaned_message])

    prediction = model.predict(message_tfidf)[0]

    print("\nPrediction Result:", prediction.upper())

    if prediction == "spam":
        print("This message is likely SPAM.")
    else:
        print("This message is likely HAM / legitimate.")

    choice = input("\nDo you want to check another message? (yes/no): ")

    if choice.lower() != "yes":
        print("\nThank you for using Email Spam Detection System.")
        break
