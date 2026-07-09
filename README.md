# Email Spam Detection with Machine Learning

## Task 4 - Data Science Internship Project

This project builds a Natural Language Processing (NLP) machine learning model to classify text messages as Spam or Ham (legitimate messages).

## Objective

The objective of this project is to preprocess text data, extract features using TF-IDF, train multiple machine learning classifiers, evaluate their performance, and build a prediction system that can classify new messages as Spam or Ham.

## Dataset

The project uses the SMS Spam Collection Dataset containing labelled text messages.

The dataset contains two main columns:

- label - Indicates whether the message is Spam or Ham
- message - Contains the text message

## Technologies Used

- Python 3.11
- Pandas
- Scikit-learn
- Matplotlib
- WordCloud
- Joblib
- IDLE

## Project Workflow

1. Load the spam dataset
2. Check dataset information and missing values
3. Remove duplicate and missing records
4. Analyze Spam and Ham class distribution
5. Preprocess the text data
6. Convert text to lowercase
7. Remove punctuation and numbers
8. Apply TF-IDF feature extraction
9. Split the dataset into training and testing sets
10. Train Multinomial Naive Bayes
11. Train Logistic Regression
12. Evaluate both models
13. Compare model performance
14. Select the best-performing model
15. Generate data visualizations
16. Save the trained model and TF-IDF vectorizer
17. Build a prediction program for new messages

## Machine Learning Models

Two machine learning algorithms were trained and evaluated:

- Multinomial Naive Bayes
- Logistic Regression

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 97.00% | 100.00% | 76.34% | 86.58% |
| Logistic Regression | 95.36% | 97.70% | 64.89% | 77.98% |

Based on the F1-score, Multinomial Naive Bayes was selected as the best-performing model.

## What is TF-IDF?

TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a text feature extraction technique that converts text data into numerical values that machine learning models can understand.

TF-IDF gives higher importance to words that are useful in a particular message while reducing the importance of words that appear frequently across many messages.

## Why is Recall Important in Spam Detection?

Recall measures how many actual spam messages are correctly identified by the model.

Recall is important because a model with low recall may fail to detect many spam messages. Undetected spam messages can contain scams, phishing links, misleading advertisements, or other harmful content.

Therefore, recall is an important evaluation metric when developing a spam detection system.

## Project Structure

Task4_Email_Spam_Detection/

    data/
        spam.csv

    images/
        class_distribution.png
        confusion_matrix.png
        spam_wordcloud.png
        ham_wordcloud.png

    models/
        spam_detection_model.pkl
        tfidf_vectorizer.pkl

    src/
        main.py
        predict.py

    README.md
    requirements.txt

## How to Run the Project

### Step 1: Install Required Libraries

Run the following command:

    pip install pandas scikit-learn matplotlib nltk wordcloud joblib

### Step 2: Train the Models

Run:

    python src/main.py

This will preprocess the dataset, train the machine learning models, evaluate their performance, generate visualizations, and save the best-performing model.

### Step 3: Run the Prediction System

Run:

    python src/predict.py

Enter a message when prompted, and the system will classify it as Spam or Ham.

## Output

The prediction system accepts a text message from the user and produces one of the following results:

- SPAM - The message is likely spam.
- HAM - The message is likely legitimate.

## Conclusion

This project demonstrates the complete process of building a Natural Language Processing classification system using Python and machine learning.

The Multinomial Naive Bayes model achieved the best performance with 97% accuracy and an F1-score of approximately 86.58%. The project successfully demonstrates text preprocessing, TF-IDF feature extraction, model training, model evaluation, visualization, model saving, and real-time prediction.

## Author

kavin .R
