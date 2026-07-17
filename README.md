# 🎬 Sentiment Analysis Using NLP and Machine Learning

## 📌 Project Overview

This project is a Sentiment Analysis Web Application developed using Python, Scikit-learn, and Streamlit. The application analyzes movie reviews and predicts whether the sentiment is Positive or Negative using Natural Language Processing (NLP) and Machine Learning techniques.

The project uses TF-IDF Vectorization for feature extraction and Logistic Regression for sentiment classification. An interactive Streamlit interface allows users to enter reviews and receive real-time predictions with confidence scores.

---

## 🚀 Features

* Sentiment Classification (Positive / Negative)
* TF-IDF Text Vectorization
* Logistic Regression Model
* Real-Time User Input Prediction
* Confidence Score Display
* Interactive Streamlit Web Interface
* Dataset Visualization using Pie Charts
* Model Serialization using Pickle

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Natural Language Processing (NLP)

---

## 📂 Project Structure

```text
Sentiment-Analysis
│
├── app.py
├── sentiment.py
├── reviews.csv
├── model.pkl
└── README.md
```

---

## ⚙️ Machine Learning Workflow

1. Load the review dataset.
2. Convert text into numerical features using TF-IDF Vectorization.
3. Split data into training and testing sets.
4. Train a Logistic Regression model.
5. Predict sentiment for new user reviews.
6. Display prediction results and confidence scores through Streamlit.

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit matplotlib nltk
```

### Run the Application

```bash
streamlit run app.py
```

or

```bash
py -m streamlit run app.py
```

---

## 📊 Sample Predictions

| Review                                  | Prediction |
| --------------------------------------- | ---------- |
| I loved this movie and its storyline    | Positive   |
| This movie was a complete waste of time | Negative   |
| Fantastic acting and direction          | Positive   |
| Terrible experience and poor screenplay | Negative   |

---

## 🎯 Future Enhancements

* Train on larger IMDb and Amazon review datasets
* Support multi-class sentiment analysis
* Deploy on Streamlit Cloud
* Add deep learning models such as LSTM and BERT
* Support sentiment analysis for social media posts

---

## 👨‍💻 Author

**Thevanathan M**

B.Tech Information Technology
Sri Sairam Engineering College

GitHub: https://github.com/Thevanathan346

---

## ⭐ Acknowledgements

This project was developed for learning and demonstrating concepts in Machine Learning, Natural Language Processing, and Web Application Development.
