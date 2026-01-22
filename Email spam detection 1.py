#  Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#  Sample dataset (10 spam + 10 ham messages)
messages = [
    "Win free money now", "Claim your prize immediately", "You won a lottery!",
    "Exclusive offer just for you", "Get your free gift card", "Limited time offer",
    "Congratulations, claim your prize", "Earn money quickly", "Free gift inside", "Claim your reward now",
    
    "Hey, are we meeting today?", "Project discussion at 5pm", "Let's have lunch tomorrow",
    "Can we talk later?", "Did you finish the report?", "See you at the meeting",
    "Lunch at 1pm?", "Don't forget the team call", "Are you coming to the event?", "Let's catch up later"
]

#  Labels: 1 = spam, 0 = ham
labels = [1]*10 + [0]*10

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