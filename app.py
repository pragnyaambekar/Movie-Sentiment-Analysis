import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Simple text preprocessing function that doesn't rely on NLTK
def simple_preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    else:
        return ""

# Apply preprocessing to review column
print("\nPreprocessing text data...")
df['processed_review'] = df['review'].apply(simple_preprocess_text)

# Check sentiment distribution
print("\nSentiment distribution:")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
print("\nSentiment distribution plot saved as 'sentiment_distribution.png'")

# Create features and target
X = df['processed_review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to features using TF-IDF
print("\nVectorizing text data...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train and evaluate Logistic Regression model (faster than SVM)
print("\nTraining Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, n_jobs=-1)
lr_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_tfidf)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Classification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=sorted(df['sentiment'].unique()),
    yticklabels=sorted(df['sentiment'].unique())
)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# Try Naive Bayes model as well (often works well for text)
print("\nTraining Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_nb = nb_model.predict(X_test_tfidf)

# Evaluate model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)

# Print results
print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
print(f"Classification Report:")
print(report_nb)

# Function to predict sentiment for new reviews
def predict_sentiment(review_text, model=lr_model if accuracy >= accuracy_nb else nb_model):
    # Preprocess the text
    processed = simple_preprocess_text(review_text)
    
    # Vectorize the text
    review_vector = tfidf_vectorizer.transform([processed])
    
    # Predict sentiment
    prediction = model.predict(review_vector)[0]
    
    return prediction

# Example usage
print("\nTesting with example reviews:")
example_reviews = [
    "This movie was amazing! I really enjoyed every moment of it.",
    "What a waste of time. Terrible acting and boring plot.",
    "It was okay, not the best but not the worst either."
]

for review in example_reviews:
    sentiment = predict_sentiment(review)
    print(f"Review: {review}")
    print(f"Predicted sentiment: {sentiment}\n")

print("Sentiment analysis complete!")