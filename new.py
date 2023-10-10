from joblib import load
import pandas as pd

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
predict_insurance_cost(22, 'male', 39, 0, 'yes', 'northwest')