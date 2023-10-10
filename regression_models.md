
# Regression Models in Machine Learning

## 1. Linear Regression
**Description:**  
Linear regression is used to predict the value of a dependent variable based on the value of one independent variable. The relationship between the variables is linear.

**When to use:**  
- Relationship between dependent and independent variable is linear.
- Data is homoscedastic (constant variance across values).

**Equation:**  
\[ y = eta_0 + eta_1 x_1 + \epsilon \]
Where:  
- \( y \) is the dependent variable.
- \( x_1 \) is the independent variable.
- \( eta_0 \) is the intercept.
- \( eta_1 \) is the slope.
- \( \epsilon \) is the error term.

## 2. Multiple Linear Regression
**Description:**  
Multiple linear regression is used when there are two or more independent variables.

**When to use:**  
- Relationship between dependent variable and multiple independent variables is linear.
- Data is homoscedastic.

**Equation:**  
\[ y = eta_0 + eta_1 x_1 + eta_2 x_2 + \dots + eta_n x_n + \epsilon \]

## 3. Polynomial Regression
**Description:**  
Polynomial regression is used when the relationship between the independent and dependent variable is curvilinear.

**When to use:**  
- Relationship between variables is not linear.
- Data shows a curvilinear trend.

**Equation:**  
\[ y = eta_0 + eta_1 x + eta_2 x^2 + \dots + eta_n x^n + \epsilon \]

## 4. Support Vector Regression (SVR)
**Description:**  
SVR is a type of Support Vector Machine (SVM) that uses the principle of maximizing the margin while limiting the margin violations.

**When to use:**  
- When linear regression doesn't fit well.
- Complex data with non-linear relationships.
- When a margin of tolerance (epsilon) is needed.

**Equation:**  
SVR has a more complex mathematical foundation, but the main idea is to find a hyperplane that best divides a dataset into classes.

## 5. Decision Tree Regression
**Description:**  
Decision tree regression creates a tree-like model to predict the value of a target variable based on several input features.

**When to use:**  
- When relationships between parameters are non-linear.
- Useful for feature importance.

## 6. Random Forest Regression
**Description:**  
Random forest regression is an ensemble method where multiple decision trees are trained and their predictions are averaged.

**When to use:**  
- When a single decision tree is overfitting.
- Complex datasets with non-linear relationships.
- Useful for feature importance.

**Note:** Random forest is built upon the decision tree model, and it tries to correct the habit of decision trees' overfitting to their training set.
