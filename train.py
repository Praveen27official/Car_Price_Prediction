import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from preprocess import load_data, preprocess

# Load data
df = load_data("data/car_data.csv")

# Preprocess
df = preprocess(df)

# Split
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/car_price_model.pkl")

print("✅ Model trained and saved successfully!")
