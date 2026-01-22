I have created a mini spam detector code by following below steps:
Imported library(sklearn) with its four modules countVectorizer,MultinomialNB model, train test split and accuracy score.
CountVectorizer converts text into numbers whereas MultinomialNB learn from the text whethers it is spam or not.
Train model learn which words are common in spam whereas accuracy score measure accuracy on test data.
Naivye model check probabiliy like free win more likely spam whweras words like meeting lunch more likely ham.
We can label 1 as spam 0 as ham.
We used an interactive loop so that user do not have to run program again and again he can as much messages he can and after finishing can use exit message.
