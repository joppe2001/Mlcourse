# **Natural Language Processing for Restaurant Reviews**

This document elucidates the process of analyzing a collection of restaurant reviews to determine their sentiment (positive or negative) using Natural Language Processing (NLP).

## **1. Libraries and Modules**

- **numpy**: Essential for numerical operations and array handling.
- **pandas**: Provides tools for data analysis and manipulation.
- **matplotlib**: Used for data visualization (though not directly used in this code).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## **2. Loading the Dataset**

The dataset, stored in 'Restaurant_Reviews.tsv', is a tab-separated file that contains reviews and their respective sentiments.

```python
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
```

- `delimiter = '\t'`: Specifies that the file is tab-separated.
- `quoting = 3`: Ignores double quotes during file loading.

## **3. Text Cleaning**

This process prepares the texts for analysis by:

1. Removing non-textual elements.
2. Converting to lowercase.
3. Splitting into individual words.
4. Removing stop words (common words that don't carry significant meaning in analysis).
5. Applying stemming (reducing words to their root form).

```python
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
```

## **4. Creating the Bag of Words Model**

The Bag of Words model represents text data in terms of a matrix of token counts. Each column corresponds to a unique word in the dataset, and each row corresponds to a review.

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
```

- `max_features = 1500`: Limits the number of columns to the 1500 most frequent words to reduce sparsity.

## **5. Splitting the Dataset**

The dataset is divided into:

- **Training Set**: Used to train the model.
- **Test Set**: Used to evaluate the model's performance.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
```

## **6. Building the Naive Bayes Classifier**

The Gaussian Naive Bayes model is trained on the Bag of Words representation of the reviews.

```python
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

## **7. Making Predictions**

Predictions are made for the reviews in the test set.

```python
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

## **8. Evaluating the Model**

### **8.1 Confusion Matrix**

The confusion matrix provides a detailed breakdown of correct and incorrect predictions.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

### **8.2 Accuracy Score**

The accuracy score offers the ratio of correctly predicted observations to the total observations.

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

---

By the end of this markdown document, readers should have a profound understanding of each step involved in analyzing restaurant reviews using Natural Language Processing.