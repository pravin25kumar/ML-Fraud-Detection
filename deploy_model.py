import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model
model_path = Path("/Users/pravinkumar/python/test/model.pkl")
model = joblib.load(model_path)

# App title
st.title("🤖 Fraud Detection App")
st.write("Enter transaction details below.")

# User inputs
step = st.number_input("Step", min_value=0, value=1)

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

amount = st.number_input("Amount", min_value=0.0, value=1000.0)
old_balance = st.number_input("Old Balance (Origin)", min_value=0.0, value=5000.0)
new_balance = st.number_input("New Balance (Origin)", min_value=0.0, value=4000.0)
old_dest = st.number_input("Old Balance (Destination)", min_value=0.0, value=2000.0)
new_dest = st.number_input("New Balance (Destination)", min_value=0.0, value=3000.0)

# Prediction
if st.button("Predict"):

    data = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": new_balance,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest
    }])

    # One-hot encoding
    data = pd.get_dummies(data, columns=["type"], dtype=int)

    # Match training features
    data = data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ Potentially Fraudulent Transaction")
    else:
        st.success("✅ Non-Fraudulent Transaction")