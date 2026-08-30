import streamlit as st
import pandas as pd
import pickle


# Load cleaned dataset
car = pd.read_csv("cleaned_car.csv")

# Load trained model
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))


st.title("🚗 Car Price Predictor")

st.write("Enter the details of your car to predict its price.")


# Company
companies = sorted(car["company"].unique())

company = st.selectbox(
    "Select Company",
    companies
)


# Models belonging to selected company
car_models = sorted(
    car[car["company"] == company]["name"].unique()
)

car_model = st.selectbox(
    "Select Model",
    car_models
)


# Year
years = sorted(
    car["year"].unique(),
    reverse=True
)

year = st.selectbox(
    "Select Year of Purchase",
    years
)


# Fuel Type
fuel_types = sorted(
    car["fuel_type"].unique()
)

fuel_type = st.selectbox(
    "Select Fuel Type",
    fuel_types
)


# Kilometers
kms_driven = st.number_input(
    "Enter Number of Kilometers travelled",
    min_value=0,
    value=10000
)


# Prediction
if st.button("Predict Price"):

    input_data = pd.DataFrame(
        [[
            car_model,
            company,
            year,
            kms_driven,
            fuel_type
        ]],
        columns=[
            "name",
            "company",
            "year",
            "kms_driven",
            "fuel_type"
        ]
    )

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Price: ₹ {prediction[0]:,.2f}"
    )