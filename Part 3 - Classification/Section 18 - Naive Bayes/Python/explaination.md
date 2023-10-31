# **Naive Bayes Classification for Social Network Ads**

This document elucidates the Naive Bayes Classification algorithm applied to a dataset containing social network ads interactions.

## **1. Libraries and Modules**

- **numpy**: Essential for numerical operations and array handling.
- **pandas**: A toolset for data manipulation and analysis.

```python
import numpy as np
import pandas as pd
```

## **2. Loading the Dataset**

The dataset, stored in a file named 'Social_Network_Ads.csv', likely contains user information and whether they interacted with a particular advertisement.

- `X`: Grabs all the feature columns except the last one.
- `y`: Retrieves the target variable, which is the last column.

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

## **3. Splitting the Dataset**

The data gets segregated into:

- **Training Set**: Used to train the machine learning model.
- **Test Set**: Used to gauge the model's performance.

The split allocates 75% of the data for training and the remaining 25% for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

- `random_state`: Ensures reproducibility by seeding the random number generator.

## **4. Feature Scaling**

Standard scaling is applied to the features so that they possess a similar scale, critical for distance-based algorithms.

- `fit_transform()`: Computes the mean and standard deviation (fitting), then scales the features (transforming).
- `transform()`: Uses the previously calculated mean and standard deviation to scale the features.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## **5. Building the Naive Bayes Classifier**

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

The `GaussianNB` is a variant of the Naive Bayes that assumes feature values are normally distributed.

```python
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

## **6. Making Predictions**

### **6.1 Predicting a New Data Point**

Here, the model predicts if a user, aged 30 with an estimated salary of 87,000, would interact with the ad.

```python
print(classifier.predict(sc.transform([[30,87000]])))
```

The features are scaled using `transform()` before the prediction.

### **6.2 Predicting Test Set Results**

Predictions are made for each user in the test set.

```python
y_pred = classifier.predict(X_test)
```

To compare predictions with actual values:

```python
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

## **7. Evaluating the Model**

### **7.1 Confusion Matrix**

The confusion matrix assesses the model's performance by detailing correctly and incorrectly predicted classifications.

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

### **7.2 Accuracy Score**

The accuracy score provides the ratio of accurately predicted observations to total observations.

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

---

By the end of this markdown document, readers should have a profound understanding of each step in constructing a Naive Bayes Classification model, excluding the data visualization (`matplotlib`) part.