import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("reviews.csv")

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["review"])
y = data["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# User input
user_review = input("Enter a review: ")

# Convert user review
user_vector = vectorizer.transform([user_review])

# Predict
result = model.predict(user_vector)

print("Sentiment:", result[0])