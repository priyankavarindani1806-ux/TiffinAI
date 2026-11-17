
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="TiffinAI Dashboard",
    page_icon="🍱",
    layout="centered"
)

st.title("🍱 TiffinAI")
st.subheader("Smart Tiffin Sales and Demand Prediction")

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

X = data.drop(columns="Tiffins")
y = data["Tiffins"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     ["Day", "Menu", "Holiday"]),
    ("num", "passthrough", ["Customers", "Previous_Orders"])
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X, y)

st.markdown("### Enter Prediction Details")

day = st.selectbox(
    "Select Day",
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
)

customers = st.number_input(
    "Expected Customers",
    min_value=1,
    max_value=500,
    value=80
)

previous_orders = st.number_input(
    "Previous Orders",
    min_value=1,
    max_value=500,
    value=78
)

menu = st.selectbox("Select Menu", ["Thali", "Veg"])

holiday = st.selectbox("Holiday?", ["No", "Yes"])

if st.button("Predict Tiffin Demand"):
    new_data = pd.DataFrame({
        "Day": [day],
        "Customers": [customers],
        "Previous_Orders": [previous_orders],
        "Menu": [menu],
        "Holiday": [holiday]
    })

    prediction = model.predict(new_data)[0]

    st.success("Prediction Completed!")

    st.metric(
        "Predicted Tiffin Demand",
        f"{round(prediction)} Tiffins"
    )

    st.metric(
        "Recommended Planning Quantity",
        f"{int(prediction + 3)} Tiffins"
    )

    st.info(
        "Planning quantity includes 3 additional tiffins "
        "as a manually selected safety buffer."
    )

st.markdown("---")
st.caption("TiffinAI | Machine Learning Project by Priyanka Varindani")
