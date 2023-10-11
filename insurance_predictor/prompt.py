import pandas as pd
from joblib import load

# Load the model and transformer
rf = load('trained_model.joblib')
ct = load('transformer.joblib')

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
    
    # Transform the input data using the loaded transformer
    transformed_input = transformer.transform(input_data)
    
    # Make prediction using the loaded model
    prediction = model.predict(transformed_input)
    
    return prediction[0]

if __name__ == "__main__":
    print("Predict Insurance Cost")
    age = int(input("Enter age: "))
    sex = input("Enter sex (male/female): ")
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter number of children: "))
    smoker = input("Are you a smoker? (yes/no): ")
    region = input("Enter region (northeast/northwest/southeast/southwest): ")

    predicted_cost = predict_insurance_cost(age, sex, bmi, children, smoker, region)
    # print the input and predicted insurance cost
    print(f"\nAge: {age}")
    print(f"\nSex: {sex}")
    print(f"\nBMI: {bmi}")
    print(f"\nChildren: {children}")
    print(f"\nSmoker: {smoker}")
    print(f"\nRegion: {region}")
    print(f"\nPredicted Insurance Cost: ${predicted_cost:.2f}")

input("Press Enter to exit...")