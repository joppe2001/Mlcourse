# **Decision Tree Classification for Social Network Ads**

This document provides a step-by-step explanation of a Decision Tree Classification algorithm applied to a dataset containing social network ads.

## **1. Libraries and Modules**

- **numpy**: This library is essential for mathematical operations and handling arrays.
- **pandas**: A powerful tool for data analysis and manipulation.

```python
import numpy as np
import pandas as pd
```

## **2. Loading the Dataset**

The dataset is loaded from a file named 'Social_Network_Ads.csv'. This dataset likely contains information about users and whether or not they clicked on a particular advertisement.

- `X`: Contains feature columns, excluding the last column (independent variables).
- `y`: Contains the target variable, which is the last column in the dataset (dependent variable).

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

## **3. Splitting the Dataset**

The dataset is divided into two subsets:

- **Training Set**: Used to train the machine learning model.
- **Test Set**: Used to evaluate the model's performance.

The `train_test_split` function is used to achieve this split, with 75% of the data for training and the remaining 25% for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

- `random_state`: Ensures reproducibility by setting a seed for the random number generator.

## **4. Feature Scaling**

This step ensures that all features have the same scale, which is crucial for most machine learning algorithms. The `StandardScaler` scales the features so that they have a mean of 0 and a standard deviation of 1.

- `fit_transform()`: Computes the mean and standard deviation (fitting) and then scales the features (transforming).
- `transform()`: Uses the previously computed mean and standard deviation to scale the features.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## **5. Building the Decision Tree Classifier**

The Decision Tree Classifier is a type of algorithm that makes decisions based on asking multiple questions. Here, the classifier is initialized using the entropy criterion, which measures the level of impurity or randomness in the data.

```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```

- `criterion='entropy'`: This aims to reduce the level of entropy (or disorder) in the dataset with each decision.
- `random_state`: Just like earlier, it ensures reproducibility.

## **6. Making Predictions**

### **6.1 Predicting a New Data Point**

The model can be used to predict whether a user of age 30 with an estimated salary of 87,000 would click on the ad or not.

```python
print(classifier.predict(sc.transform([[30,87000]])))
```

- The features are scaled using the `transform()` method before prediction.

### **6.2 Predicting the Test Set Results**

For each user in the test set, the model predicts if they clicked on the ad or not.

```python
y_pred = classifier.predict(X_test)
```

The results can be viewed side by side with the actual values:

```python
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

## **7. Evaluating the Model**

### **7.1 Confusion Matrix**

A confusion matrix is used to evaluate the performance of classification models. It provides details on correctly and incorrectly predicted classifications.

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

### **7.2 Accuracy Score**

This provides a ratio of correctly predicted observations to the total observations.

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

---

By the end of this document, readers should have a comprehensive understanding of each step involved in building a Decision Tree Classification model, excluding the data visualization (matplotlib) part.