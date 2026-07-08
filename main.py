# ==========================================================
# TASK 4 - EMAIL SPAM DETECTION WITH MACHINE LEARNING
# ==========================================================

# Import required libraries
import os
import re
import string
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================================
# 1. CREATE FILE PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create folders if they do not exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==========================================================
# 2. LOAD THE DATASET
# ==========================================================

print("=" * 60)
print("        EMAIL SPAM DETECTION USING MACHINE LEARNING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================================
# 3. DATA CLEANING
# ==========================================================

# Keep only required columns
df = df[["label", "message"]]

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)


# ==========================================================
# 4. CHECK CLASS DISTRIBUTION
# ==========================================================

print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nClass Distribution Percentage:")
print(df["label"].value_counts(normalize=True) * 100)


# Create class distribution chart

class_counts = df["label"].value_counts()

plt.figure(figsize=(6, 5))

class_counts.plot(kind="bar")

plt.title("Spam vs Ham Message Distribution")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(IMAGES_DIR, "class_distribution.png")
)

plt.show()


# ==========================================================
# 5. TEXT PREPROCESSING
# ==========================================================

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


print("\nCleaning messages...")

df["cleaned_message"] = df["message"].apply(clean_text)

print("\nOriginal and Cleaned Messages:")

print(
    df[["message", "cleaned_message"]].head()
)


# ==========================================================
# 6. CREATE WORD CLOUDS
# ==========================================================

print("\nCreating WordCloud visualizations...")

spam_messages = " ".join(
    df[df["label"] == "spam"]["cleaned_message"]
)

ham_messages = " ".join(
    df[df["label"] == "ham"]["cleaned_message"]
)


# Spam WordCloud

spam_wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(spam_messages)

plt.figure(figsize=(10, 5))

plt.imshow(spam_wordcloud)

plt.axis("off")

plt.title("Most Common Words in Spam Messages")

plt.tight_layout()

plt.savefig(
    os.path.join(IMAGES_DIR, "spam_wordcloud.png")
)

plt.show()


# Ham WordCloud

ham_wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(ham_messages)

plt.figure(figsize=(10, 5))

plt.imshow(ham_wordcloud)

plt.axis("off")

plt.title("Most Common Words in Ham Messages")

plt.tight_layout()

plt.savefig(
    os.path.join(IMAGES_DIR, "ham_wordcloud.png")
)

plt.show()


# ==========================================================
# 7. DEFINE FEATURES AND TARGET
# ==========================================================

X = df["cleaned_message"]

y = df["label"]


# ==========================================================
# 8. TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("\nTraining Data Size:", len(X_train))

print("Testing Data Size:", len(X_test))


# ==========================================================
# 9. TF-IDF FEATURE EXTRACTION
# ==========================================================

print("\nApplying TF-IDF Vectorization...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF transformation completed!")

print("Number of TF-IDF Features:")

print(X_train_tfidf.shape[1])


# ==========================================================
# 10. MODEL 1 - MULTINOMIAL NAIVE BAYES
# ==========================================================

print("\n" + "=" * 60)

print("MULTINOMIAL NAIVE BAYES MODEL")

print("=" * 60)

nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)

nb_predictions = nb_model.predict(X_test_tfidf)


nb_accuracy = accuracy_score(
    y_test,
    nb_predictions
)

nb_precision = precision_score(
    y_test,
    nb_predictions,
    pos_label="spam"
)

nb_recall = recall_score(
    y_test,
    nb_predictions,
    pos_label="spam"
)

nb_f1 = f1_score(
    y_test,
    nb_predictions,
    pos_label="spam"
)


print("\nAccuracy :", round(nb_accuracy, 4))

print("Precision:", round(nb_precision, 4))

print("Recall   :", round(nb_recall, 4))

print("F1 Score :", round(nb_f1, 4))


# ==========================================================
# 11. MODEL 2 - LOGISTIC REGRESSION
# ==========================================================

print("\n" + "=" * 60)

print("LOGISTIC REGRESSION MODEL")

print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = lr_model.predict(
    X_test_tfidf
)


lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

lr_precision = precision_score(
    y_test,
    lr_predictions,
    pos_label="spam"
)

lr_recall = recall_score(
    y_test,
    lr_predictions,
    pos_label="spam"
)

lr_f1 = f1_score(
    y_test,
    lr_predictions,
    pos_label="spam"
)


print("\nAccuracy :", round(lr_accuracy, 4))

print("Precision:", round(lr_precision, 4))

print("Recall   :", round(lr_recall, 4))

print("F1 Score :", round(lr_f1, 4))


# ==========================================================
# 12. COMPARE MODELS
# ==========================================================

results = pd.DataFrame({

    "Model": [
        "Multinomial Naive Bayes",
        "Logistic Regression"
    ],

    "Accuracy": [
        nb_accuracy,
        lr_accuracy
    ],

    "Precision": [
        nb_precision,
        lr_precision
    ],

    "Recall": [
        nb_recall,
        lr_recall
    ],

    "F1 Score": [
        nb_f1,
        lr_f1
    ]

})


print("\n" + "=" * 60)

print("MODEL COMPARISON")

print("=" * 60)

print(results)


# ==========================================================
# 13. SELECT BEST MODEL
# ==========================================================

if nb_f1 >= lr_f1:

    best_model = nb_model

    best_predictions = nb_predictions

    best_model_name = "Multinomial Naive Bayes"

else:

    best_model = lr_model

    best_predictions = lr_predictions

    best_model_name = "Logistic Regression"


print("\nBest Model:", best_model_name)


# ==========================================================
# 14. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["ham", "spam"]
)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "Ham",
        "Spam"
    ]
)

display.plot()

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        IMAGES_DIR,
        "confusion_matrix.png"
    )
)

plt.show()


# ==========================================================
# 15. SAVE BEST MODEL AND TF-IDF VECTORIZER
# ==========================================================

model_path = os.path.join(
    MODELS_DIR,
    "spam_detection_model.pkl"
)

vectorizer_path = os.path.join(
    MODELS_DIR,
    "tfidf_vectorizer.pkl"
)


joblib.dump(
    best_model,
    model_path
)

joblib.dump(
    tfidf,
    vectorizer_path
)


print("\nModel saved successfully!")

print("Saved Model:")

print(model_path)

print("\nTF-IDF Vectorizer saved successfully!")

print("Saved Vectorizer:")

print(vectorizer_path)


# ==========================================================
# 16. PROJECT COMPLETED
# ==========================================================

print("\n" + "=" * 60)

print("EMAIL SPAM DETECTION PROJECT COMPLETED SUCCESSFULLY")

print("=" * 60)
