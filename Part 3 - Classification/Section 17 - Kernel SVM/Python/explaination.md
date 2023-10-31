# **Kernel SVM Classification for Social Network Ads**

This document elucidates the Kernel Support Vector Machine (SVM) Classification algorithm applied to a dataset containing social network ads interactions.

## **1. Libraries and Modules**

- **numpy**: Essential for numerical operations and array handling.
- **pandas**: Offers tools for data manipulation and analysis.

```python
import numpy as np
import pandas as pd
```

## **2. Loading the Dataset**

The dataset, stored in 'Social_Network_Ads.csv', likely consists of user information and whether they interacted with a particular advertisement.

- `X`: Retrieves all the feature columns except the last one.
- `y`: Contains the target variable, which is the last column.

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

## **3. Splitting the Dataset**

The dataset is divided into:

- **Training Set**: Used to train the machine learning model.
- **Test Set**: Used to evaluate the model's performance.

A 75% allocation of the data is used for training, and the remaining 25% is reserved for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

- `random_state`: Ensures reproducibility by setting a seed for the random number generator.

## **4. Feature Scaling**

Standard scaling ensures all features have a similar scale, which is crucial for distance-based algorithms like SVM.

- `fit_transform()`: Computes the mean and standard deviation (fitting) and then scales the features (transforming).
- `transform()`: Uses the previously computed mean and standard deviation to scale the features.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## **5. Building the Kernel SVM Classifier**

Kernel SVM introduces the kernel trick to transform input data into a higher-dimensional space to find a hyperplane that best separates the classes. The Radial Basis Function (RBF) kernel is used here.

```python
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
```

## **6. Making Predictions**

### **6.1 Predicting a New Data Point**

The model predicts if a 30-year-old user with an estimated salary of 87,000 would interact with the ad.

```python
print(classifier.predict(sc.transform([[30,87000]])))
```

The features are scaled using `transform()` before prediction.

### **6.2 Predicting Test Set Results**

Predictions are made for every user in the test set.

```python
y_pred = classifier.predict(X_test)
```

Comparing predictions with actual values:

```python
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

## **7. Evaluating the Model**

### **7.1 Confusion Matrix**

The confusion matrix evaluates the model's performance by detailing the correctly and incorrectly predicted classifications.

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

### **7.2 Accuracy Score**

This score offers the ratio of correctly predicted observations to the total observations.

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

---

By the end of this markdown document, readers should have a profound understanding of each step in constructing a Kernel SVM Classification model, excluding the data visualization (`matplotlib`) part.