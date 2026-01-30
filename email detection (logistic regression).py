import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("email_spam_dataset.csv", encoding="utf-8")

# 3. Keep only required columns
df = df[['label', 'email_text']]

# 4. Convert labels to numbers
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['email_text'],
    df['label'],
    test_size=0.25,
    random_state=42
)

# 6. Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 8. Make predictions
y_pred = model.predict(X_test_vec)

# 9. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Predict custom email
def predict_email(text):
    text_vec = vectorizer.transform([text])
    prob = model.predict_proba(text_vec)[0][1]
    return f"SPAM ({prob:.2f})" if prob > 0.5 else f"HAM ({1-prob:.2f})"


# 11. Interactive prediction loop
print("\n=== Interactive Spam Detector ===")
print("Type 'exit' to quit.")
while True:
    user_input = input("Enter an email to test: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    print(predict_email(user_input))