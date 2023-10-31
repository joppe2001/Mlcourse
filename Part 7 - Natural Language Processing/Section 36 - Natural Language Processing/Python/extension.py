import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('combined_prompts.csv')
X = dataset['prompt'].values
y_html = dataset['html'].values
y_css = dataset['css'].values

# Cleaning the prompts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, len(X)):
    prompt = re.sub('[^a-zA-Z]', ' ', X[i])
    prompt = prompt.lower()
    prompt = prompt.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    prompt = [ps.stem(word) for word in prompt if not word in set(all_stopwords)]
    prompt = ' '.join(prompt)
    corpus.append(prompt)

# Creating the Bag of Words model with reduced features
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus)  # No .toarray() here to keep it as a sparse matrix

# Splitting the dataset into the Training set and Test set for HTML prediction
from sklearn.model_selection import train_test_split
X_train_html, X_test_html, y_train_html, y_test_html = train_test_split(X, y_html, test_size = 0.20, random_state = 0)

# Training the Random Forest model on the Training set for HTML prediction with reduced trees and depth
from sklearn.ensemble import RandomForestClassifier
classifier_html = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=0)
classifier_html.fit(X_train_html, y_train_html)

# Splitting the dataset into the Training set and Test set for CSS prediction
X_train_css, X_test_css, y_train_css, y_test_css = train_test_split(X, y_css, test_size = 0.20, random_state = 0)

# Training the Random Forest model on the Training set for CSS prediction
classifier_css = RandomForestClassifier(n_estimators=100, random_state=0)
classifier_css.fit(X_train_css, y_train_css)

# Function to predict based on user input
def predict_html_css(user_prompt):
    user_prompt = re.sub('[^a-zA-Z]', ' ', user_prompt)
    user_prompt = user_prompt.lower().split()
    user_prompt = [ps.stem(word) for word in user_prompt if not word in set(all_stopwords)]
    user_input = cv.transform([' '.join(user_prompt)]).toarray()
    html_prediction = classifier_html.predict(user_input)
    css_prediction = classifier_css.predict(user_input)
    return html_prediction[0], css_prediction[0]

# Test the function
user_prompt = input("Enter your design prompt: ")
predicted_html, predicted_css = predict_html_css(user_prompt)
print("\nPredicted HTML:\n", predicted_html)
print("\nPredicted CSS:\n", predicted_css)
