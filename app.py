import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix



df = pd.read_csv("dataset.csv")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)

# UI
st.title("Let Me Classify")
st.write("Multi-Category Text Classification using Naive Bayes")

text_input = st.text_area("Enter text to classify")

if st.button("Classify"):

    if text_input.strip() == "":
        st.warning("Please enter some text to classify.")

    else:
        vec = vectorizer.transform([text_input])

        prediction = model.predict(vec)
        prob = model.predict_proba(vec)

        st.success(f"Predicted Category: {prediction[0]}")

        st.write("Class Probabilities:")

        prob_df = pd.DataFrame({
            "Category": model.classes_,
            "Probability": prob[0]
        })

        st.bar_chart(prob_df.set_index("Category"))

