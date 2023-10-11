
#  load saved model 
from joblib import load
from keras.models import load_model
import pandas as pd
import numpy as np

loaded_model = load_model('trainedModel')
loaded_transformer = load('transformer.joblib')

def predict_cardiovascular(age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, glucose, smoker, alcoholic, bmi, active, bp_category, model=loaded_model, transformer=loaded_transformer):

    age = age / 365.25
    
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
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
    
    input_data.replace({-1: np.nan}, inplace=True)
    
    transformed_input = transformer.transform(input_data)
    
    prediction = model.predict(transformed_input)
    
    return prediction[0]

def predict_cardiovascular_from_input(model=loaded_model, transformer=loaded_transformer):
    age = int(input("Enter age: "))
    gender = int(input("Enter gender (0 for female, 1 for male): "))
    height = float(input("Enter height in cm: "))
    weight = float(input("Enter weight in kg: "))
    systolic_bp = int(input("Enter systolic blood pressure in mmHg: "))
    diastolic_bp = int(input("Enter diastolic blood pressure in mmHg: "))
    cholesterol = int(input("Enter cholesterol level (1 for normal, 2 for above normal): "))
    glucose = int(input("Enter glucose level (1 for normal, 2 for above normal): "))
    smoker = int(input("Do you smoke? (0 for no, 1 for yes): "))
    alcoholic = int(input("Do you drink alcohol? (0 for no, 1 for yes): "))
    bmi = weight / ((height/100)**2)
    active = int(input("Enter activity level (0 for not active, 1 for moderately active, 2 for very active): "))
    bp_category = input("Enter blood pressure category (Normal, Elevated, Hypertension Stage 1, Hypertension Stage 2): ")
    
    prediction = predict_cardiovascular(age, gender, height, weight, systolic_bp, diastolic_bp, cholesterol, glucose, smoker, alcoholic, bmi, active, bp_category, model, transformer)
    
    return prediction


prediction = predict_cardiovascular_from_input()

if prediction[0] == 1:
    print('You might have cardiovascular disease!')
else:
    print('You are healthy!')