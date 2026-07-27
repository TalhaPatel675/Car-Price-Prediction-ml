import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction")
st.write("Enter the feature values below to predict the car price.")

# ---------------------------------
# Load Model & Feature Names
# ---------------------------------
try:
    model = joblib.load("models/linear_regression.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run train.py first.")
    st.stop()

# ---------------------------------
# User Inputs
# ---------------------------------
input_data = {}

st.subheader("Enter Feature Values")

for feature in feature_names:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0,
        format="%.2f"
    )

# ---------------------------------
# Predict
# ---------------------------------
if st.button("Predict Price"):

    input_df = pd.DataFrame([input_data])

    # Ensure the feature order matches the model
    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")
