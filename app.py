import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("reviews.csv")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("📊 Dataset Information")

st.sidebar.write(f"Total Reviews: {len(data)}")

st.sidebar.write("Sentiment Distribution")

st.sidebar.write(data["sentiment"].value_counts())

# ----------------------------
# Pie Chart
# ----------------------------
st.subheader("📈 Dataset Distribution")

sentiment_counts = data["sentiment"].value_counts()

fig, ax = plt.subplots()

ax.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%"
)

st.pyplot(fig)

# ----------------------------
# Text Vectorization
# ----------------------------
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["review"])
y = data["sentiment"]

# ----------------------------
# Train Model
# ----------------------------
model = LogisticRegression()

model.fit(X, y)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

# ----------------------------
# Main Title
# ----------------------------
st.title("🎬 Movie Review Sentiment Analyzer")

st.markdown("""
Analyze movie reviews using **Machine Learning** and **Natural Language Processing (NLP)**.
""")

# ----------------------------
# Examples
# ----------------------------
st.subheader("📝 Example Reviews")

st.write("😊 Positive: I loved the acting and storyline")
st.write("😊 Positive: This movie was fantastic and entertaining")

st.write("😞 Negative: This movie was a complete waste of time")
st.write("😞 Negative: Worst movie I have ever watched")

# ----------------------------
# User Input
# ----------------------------
review = st.text_area(
    "Enter a Movie Review",
    placeholder="Type your review here..."
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        user_vector = vectorizer.transform([review])

        prediction = model.predict(user_vector)[0]

        probability = model.predict_proba(user_vector)

        confidence = max(probability[0]) * 100

        st.subheader("Prediction Result")

        if prediction == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.progress(int(confidence))

        st.write(f"Confidence Score: {confidence:.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "Built using Python, Scikit-learn, Streamlit, TF-IDF, and Logistic Regression."
)