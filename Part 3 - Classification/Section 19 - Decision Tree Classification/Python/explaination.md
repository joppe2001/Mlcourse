# Decision Tree Classification for Social Network Ads

This document provides a detailed walkthrough of a Python script that uses a Decision Tree Classifier to classify users based on their social network ad interactions.

## **1. Libraries and Modules**

- **numpy**: Handles arrays and mathematical operations.
- **matplotlib.pyplot**: Provides functionalities for plotting and visualizing data.
- **pandas**: Used for data manipulation and analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## **2. Loading the Dataset**

The data is loaded from a CSV file named 'Social_Network_Ads.csv'. The features (independent variables) are stored in `X` and the target (dependent variable) in `y`.

```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

## **3. Splitting the Dataset**

The data is divided into training and testing subsets using the `train_test_split` method. 75% of the data is used for training and 25% for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

## **4. Feature Scaling**

Standard scaling is applied to the features so that they all have a similar scale. This is important for algorithms that rely on distances or gradients.

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## **5. Building the Decision Tree Classifier**

The Decision Tree Classifier is initialized with the entropy criterion and then trained on the training data.

```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```

## **6. Making Predictions**

A new data point is predicted to see the output of the trained model. Additionally, predictions are made for the entire test set.

```python
print(classifier.predict(sc.transform([[30,87000]])))
y_pred = classifier.predict(X_test)
```

## **7. Evaluating the Model**

A confusion matrix and accuracy score are calculated to evaluate the model's performance on the test set.

```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

## **8. Visualizing the Results**

Visual representations of the classifier's performance on the training and test sets are created using `matplotlib`.

### **8.1 Training Set Visualization**

```python
from matplotlib.colors import ListedColormap
...
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

### **8.2 Test Set Visualization**

```python
from matplotlib.colors import ListedColormap
...
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

---

This markdown document provides a comprehensive and detailed understanding of the Decision Tree Classification code, with explanations for each major step.