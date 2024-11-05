# Simple Retail Turnover Prediction App

import streamlit as st
import pandas as pd
from joblib import load

# Load the saved model
model = load('tuned_ridge_model.pkl')

# App title
st.title("Retail Turnover Prediction")

# Main area for input parameters
st.header("Input Information")

# Get user input for the model
hours_own = st.number_input("Hours Owned", min_value=0, step=1, format="%d")
sales_units = st.number_input("Sales Units", min_value=0, step=1, format="%d")

# Create a dataframe for prediction
input_data = pd.DataFrame({
    "HoursOwn": [hours_own],
    "Sales units": [sales_units]
})

# Predict turnover
if st.button("Predict Turnover"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Turnover Amount: ${prediction[0]:,.2f}")  # Formatting for better readability
    except Exception as e:
        st.error(f"Error in prediction: {e}")
