## Importing necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
    
## Convert sample data to a pandas DataFrame
data = pd.read_csv('cardio_data_processed.csv')
## Extract features and target
# extract features and target
X = data.drop(['id', 'cardio', 'age_years', 'bp_category_encoded' ], axis=1)
y = data['cardio']
    
## Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

## Create the column transformer
ct = ColumnTransformer([
    ('one_hot_encoder', OneHotEncoder(drop='first'), [ 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bp_category']),
    ('standard_scaler', StandardScaler(), ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']),
], remainder='passthrough')
## Transform training and test data

X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)

    
## Create the random forest regression model

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train_transformed.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train_transformed, y_train, epochs=50, batch_size=300, verbose=0)

y_pred = model.predict(X_test_transformed)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 score:', r2_score(y_test, y_pred))

## Evaluation metrics
# Evaluate the model's performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 score:', r2_score(y_test, y_pred))
    
## Save and load model and transformer

from joblib import dump, load

# After training your model and transformer:
model.save('trainedModel')
dump(ct, 'transformer.joblib')

from keras.models import load_model
loaded_model = load_model('trainedModel')
loaded_transformer = load('transformer.joblib')
    
## predict_insurance_cost function
def predict_cardiovascular(age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, glucose, smoker, alcoholic, bmi, active, bp_category, model=loaded_model, transformer=loaded_transformer):
    # Convert age from days to years
    age = age / 365.25
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender if gender == 1 else 0],
        'height': [height],
        'weight': [weight],
        'ap_hi': [systolic_bp],
        'ap_lo': [diastolic_bp],
        'cholesterol': [cholesterol],
        'gluc': [glucose],
        'smoke': [smoker],
        'alco': [alcoholic],
        'bmi': [bmi],
        'active': [active],
        'bp_category': [bp_category]
    })
    
    # Replace unknown categories with NaN
    input_data.replace({-1: np.nan}, inplace=True)
    
    # Transform the input data using the same transformer used during training
    transformed_input = transformer.transform(input_data)
    
    # Make prediction using the trained model
    prediction = model.predict(transformed_input)
    
    # Return the predicted insurance cost
    return prediction[0]

## Test the function with a sample input

age = 22 * 365.25  # age in days
gender = 0
height = 184
weight = 67
systolic_bp = 100
diastolic_bp = 80
cholesterol = 1
glucose = 1
smoker = 0
alcoholic = 1
bmi = weight / ((height/100)**2)
active = 1
bp_category = 'Hypertension Stage 2'

prediction = predict_cardiovascular(age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, glucose, smoker, alcoholic, bmi, active, bp_category)

if prediction == 1:
    print('You might have cardiovascular disease!')
else:
    print('You are healthy!')
#  create a function that prompts a user to give their information and returns the predicted  predict_cardiovascular function

def predict_cardiovascular_from_input():
    # Prompt user to enter their information
    age = float(input("Enter your age in years: "))
    gender = int(input("Enter your gender (0 for female, 1 for male): "))
    height = float(input("Enter your height in cm: "))
    weight = float(input("Enter your weight in kg: "))
    systolic_bp = float(input("Enter your systolic blood pressure in mmHg: "))
    diastolic_bp = float(input("Enter your diastolic blood pressure in mmHg: "))
    cholesterol = int(input("Enter your cholesterol level (1 for normal, 2 for above normal): "))
    glucose = int(input("Enter your glucose level (1 for normal, 2 for above normal): "))
    smoker = int(input("Do you smoke? (0 for no, 1 for yes): "))
    alcoholic = int(input("Do you drink alcohol? (0 for no, 1 for yes): "))
    active = int(input("How active are you? (0 for not active, 1 for moderately active, 2 for very active): "))
    bp_category = input("Enter your blood pressure category (Normal, Elevated, Hypertension Stage 1, Hypertension Stage 2): ")
    
    # Convert age from years to days
    age = age * 365.25
    
    # Calculate BMI
    bmi = weight / ((height/100)**2)
    
    # Make prediction using the trained model
    prediction = predict_cardiovascular(age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, glucose, smoker, alcoholic, bmi, active, bp_category)
    
    # Return the predicted cardiovascular disease risk
    return prediction