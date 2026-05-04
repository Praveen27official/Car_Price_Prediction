import joblib
import pandas as pd

model = joblib.load("models/car_price_model.pkl")

def predict_price(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]

# Example usage
sample = {
    'Present_Price': 5.59,
    'Kms_Driven': 27000,
    'Owner': 0,
    'Car_Age': 5,
    'Fuel_Type_Diesel': 0,
    'Fuel_Type_Petrol': 1,
    'Seller_Type_Individual': 1,
    'Transmission_Manual': 1
}

print("Predicted Price:", predict_price(sample))
