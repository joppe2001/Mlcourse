# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Convert sample data to a pandas DataFrame
data = pd.read_csv('insurance.csv')

# Proceeding with the outlined steps
# Extract features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

# Create the column transformer
ct = ColumnTransformer([
    ('one_hot_encoder', OneHotEncoder(), ['sex', 'smoker', 'region']),
    ('standard_scaler', StandardScaler(), ['age', 'bmi', 'children'])
], remainder='passthrough')

# Transform training and test data
X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)

# Create the random forest regression model  
rf = RandomForestRegressor(n_estimators=100, random_state=1)

# Fit the model to the training data
rf.fit(X_train_transformed, y_train)

# Predict the test data
y_pred = rf.predict(X_test_transformed)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
score = rf.score(X_test_transformed, y_test)

score, mae, rmse, y_pred[:5], y_test[:5].values

from joblib import dump

# After training your model and transformer:
dump(rf, 'trained_model.joblib')
dump(ct, 'transformer.joblib')

from joblib import load

# Load the model and transformer
rf = load('trained_model.joblib')
ct = load('transformer.joblib')

# Now, you can use `rf` and `ct` in your predict_insurance_cost function without needing the dataset.


def predict_insurance_cost(age, sex, bmi, children, smoker, region, model=rf, transformer=ct):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Transform the input data using the same transformer used during training
    transformed_input = transformer.transform(input_data)
    
    # Make prediction using the trained model
    prediction = model.predict(transformed_input)
    
    return prediction[0]

# Test the function with a sample input
predict_insurance_cost(22, 'male', 19, 0, 'no', 'northwest')

