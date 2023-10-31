import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Load saved models and count vectorizer
from joblib import load

classifier_html = load('classifier_html.pkl')
classifier_css = load('classifier_css.pkl')
cv = load('count_vectorizer.pkl')

# Download stopwords
nltk.download('stopwords')

# Function to predict based on user input
def predict_html_css(user_prompt):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    user_prompt = re.sub('[^a-zA-Z]', ' ', user_prompt)
    user_prompt = user_prompt.lower().split()
    user_prompt = [ps.stem(word) for word in user_prompt if not word in set(all_stopwords)]
    user_input = cv.transform([' '.join(user_prompt)])
    html_prediction = classifier_html.predict(user_input)
    css_prediction = classifier_css.predict(user_input)
    return html_prediction[0], css_prediction[0]

# Test the function
user_prompt = input("Enter your design prompt: ")
predicted_html, predicted_css = predict_html_css(user_prompt)
print("\nPredicted HTML:\n", predicted_html)
print("\nPredicted CSS:\n", predicted_css)
