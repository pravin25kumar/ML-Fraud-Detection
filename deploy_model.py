
import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
import base64

# -------------------------
# 1. File paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
image_path = BASE_DIR / "background.jpg"
model_path = BASE_DIR / "model.pkl"

# -------------------------
# 2. Page configuration
# -------------------------
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🤖",
    layout="wide"
)


# -------------------------
# 3. Background and styling
# -------------------------
if image_path.exists():
    image_base64 = base64.b64encode(
        image_path.read_bytes()
    ).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(234, 244, 255, 0.78),
                    rgba(214, 233, 255, 0.78)
                ),
                url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Dark title */
        h1 {{
            color: #102A43 !important;
            font-weight: 800 !important;
        }}

        /* Dark subtitle */
        .stApp p {{
            color: #243B53 !important;
        }}

        /* Blue Predict button with white text */
        .stButton > button {{
            background-color: #1976D2 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
        }}

        /* Hover effect */
        .stButton > button:hover {{
            background-color: #1259A7 !important;
            color: #FFFFFF !important;
        }}
        </style>
    """, unsafe_allow_html=True)

else:
    st.warning("background.jpg not found. Check your project folder.")

# -------------------------
# 3. Background and styling
# -------------------------
if image_path.exists():
    image_base64 = base64.b64encode(
        image_path.read_bytes()
    ).decode()

    st.markdown(f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(234, 244, 255, 0.78),
                    rgba(214, 233, 255, 0.78)
                ),
                url("data:image/jpeg;base64,{image_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Dark title */
        h1 {{
            color: #102A43 !important;
            font-weight: 800 !important;
        }}

        /* Dark subtitle */
        .stApp p {{
            color: #243B53 !important;
        }}

        /* Blue Predict button with white text */
        .stButton > button {{
            background-color: #1976D2 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
        }}

        /* Hover effect */
        .stButton > button:hover {{
            background-color: #1259A7 !important;
            color: #FFFFFF !important;
        }}
        </style>
    """, unsafe_allow_html=True)

else:
    st.warning("background.jpg not found. Check your project folder.")


# -------------------------
# 4. Load model
# -------------------------
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# -------------------------
# 5. App title
# -------------------------
st.title("🤖 Fraud Detection App")
st.write("Enter transaction details below.")

# -------------------------
# 6. User inputs
# -------------------------
step = st.number_input("Step", min_value=0, value=1)

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

old_balance = st.number_input(
    "Old Balance (Origin)",
    min_value=0.0,
    value=5000.0
)

new_balance = st.number_input(
    "New Balance (Origin)",
    min_value=0.0,
    value=4000.0
)

old_dest = st.number_input(
    "Old Balance (Destination)",
    min_value=0.0,
    value=2000.0
)

new_dest = st.number_input(
    "New Balance (Destination)",
    min_value=0.0,
    value=3000.0
)

# -------------------------
# 7. Prediction
# -------------------------
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
    data = pd.get_dummies(
        data,
        columns=["type"],
        dtype=int
    )

    # Match model training features
    if not hasattr(model, "feature_names_in_"):
        st.error(
            "Model feature names are missing. "
            "Check how the model was trained."
        )
        st.stop()

    data = data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    try:
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.error(
                "⚠️ Potentially Fraudulent Transaction"
            )
        else:
            st.success(
                "✅ Non-Fraudulent Transaction"
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
