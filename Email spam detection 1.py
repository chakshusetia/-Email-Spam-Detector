#  Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#load dataset
df = pd.read_csv("email_spam_dataset.csv", encoding="utf-8")

# Extract messages and labels
messages = df["email_text"]

# Map 'ham' to 0, 'spam' to 1
labels = df["label"].map({"ham": 0, "spam": 1})

#  Convert text messages into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

#  Train Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

#  Interactive loop
print("=== Mini Spam Detector ===")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nEnter a message: ")
    if user_input.lower() == "exit":
        print("Goodbye! 👋")
        break

    # Convert user input to numbers
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    print("Prediction:", "Spam 🚫" if prediction == 1 else "Not Spam ✅")


    print("Prediction:", "Spam 🚫" if prediction == 1 else "Not Spam ✅")
