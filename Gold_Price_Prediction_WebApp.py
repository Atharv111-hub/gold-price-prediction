# Gold_Price_Prediction_WebApp.py

import streamlit as st
from gold_price_model import gold_price_prediction

# Streamlit Page Configuration
st.set_page_config(page_title="Gold Price Prediction", layout="centered")

# App Title
st.title("💰 Gold Price Prediction Web App")

st.write("Enter the required values below to predict the Gold Price:")

# User Input Fields — (Adjust if your model has different features)
SPX = st.number_input("S&P 500 INDEX", min_value=0.0, format="%.2f")
USO = st.number_input("UNITED STATE OIL FUND", min_value=0.0, format="%.2f")
SLV = st.number_input("iSHARE SILVER TRUST", min_value=0.0, format="%.2f")
EUR_USD = st.number_input("EURO/US DOLLER CURRENCY PAIR", min_value=0.0, format="%.2f")

# When the user clicks the Predict button
if st.button("Predict Gold Price"):
    # Use only 4 features (assuming the model was trained with 4 features)
    input_features = [SPX, USO, SLV, EUR_USD]
    
    try:
        prediction = gold_price_prediction(input_features)
        st.success(f"📈 Predicted Gold Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
