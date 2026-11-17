import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample dataset
data = pd.DataFrame({
    "Day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun",
            "Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
    "Customers": [76,81,69,88,94,86,80,75,83,71,90,96,89,82],
    "Previous_Orders": [74,79,67,84,91,82,78,73,80,69,86,93,85,80],
    "Menu": ["Thali","Veg","Thali","Veg","Thali","Veg","Thali",
             "Veg","Thali","Veg","Thali","Veg","Thali","Veg"],
    "Holiday": ["No"] * 14,
    "Tiffins": [78,82,70,89,96,84,80,77,84,72,91,98,87,83]
})

# Input and target
X = data.drop(columns="Tiffins")
y = data["Tiffins"]

# Features
categorical_features = ["Day", "Menu", "Holiday"]
numeric_features = ["Customers", "Previous_Orders"]

# Convert categorical data
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
])

# Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train model
model.fit(X, y)

# New data for prediction
new_data = pd.DataFrame({
    "Day": ["Thu"],
    "Customers": [81],
    "Previous_Orders": [79],
    "Menu": ["Veg"],
    "Holiday": ["No"]
})

# Prediction
prediction = model.predict(new_data)[0]

print("Predicted Tiffin Demand:", round(prediction))
print("Planning Quantity:", round(prediction) + 3) se
