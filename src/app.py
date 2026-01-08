import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. SETUP: THE "GLUE" ---
# We define the cleaner class here so the saved pipeline loads without error.
class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
        return X_copy

# Cache the model so the app doesn't reload it every time you move a slider
@st.cache_resource
def load_brain():
    model = joblib.load("models/xgboost_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

# Load the artifacts
try:
    model, preprocessor = load_brain()
    print("✅ Model loaded successfully")
except Exception as e:
    st.error(f"CRITICAL ERROR: Could not load model. {e}")
    st.stop()

# --- 2. THE SIDEBAR (CONTROLS) ---
st.set_page_config(page_title="Churn Predictor", page_icon="🛡️")

st.sidebar.header("Customer Profile")
st.sidebar.write("Adjust sliders to simulate a customer.")

# INPUTS: We translate "English" sliders into the dataframe the model needs
tenure = st.sidebar.slider("Tenure (Months Stayed)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Bill ($)", 18.0, 120.0, 70.0)

# Auto-calculate Total Charges (Logic: Bill * Months)
total_charges = monthly_charges * tenure 

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# We create the Dataframe exactly how the model expects it
# (We set other minor features to 'No' for this demo to keep it simple)
input_data = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [tenure],
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': [internet],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': [contract],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': [payment],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [str(total_charges)] # Send as string to prove our pipeline works!
})

# --- 3. THE BRAIN (PREDICTION) ---
# Clean & Transform data
X_processed = preprocessor.transform(input_data)

# Predict Probability (0 to 1)
probability = model.predict_proba(X_processed)[0][1]

# Make Decision based on OUR threshold (0.32)
# We catch churners early!
is_churn = probability > 0.32

# --- 4. THE MAIN SCREEN (DISPLAY) ---
st.title("🛡️ Customer Retention System")
st.markdown("### Real-time Risk Analysis")

# Create two columns for a dashboard look
col1, col2 = st.columns(2)

with col1:
    st.write("## Risk Score")
    # Dynamic Color: Green if safe, Red if risky
    color = "red" if is_churn else "green"
    st.markdown(f"""
        <div style='text-align: center; color: {color}; font-size: 80px; font-weight: bold;'>
        {probability:.1%}
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.write("## Decision")
    if is_churn:
        st.error("⚠️ HIGH RISK: CHURN LIKELY")
        st.write(f"**Reason:** Probability ({probability:.2f}) > Threshold (0.32)")
        st.info("💡 **Recommendation:** Offer 12-Month Contract with 15% Discount.")
    else:
        st.success("✅ SAFE CUSTOMER")
        st.write("**Status:** Customer is happy.")
        st.write("💡 **Recommendation:** Do not disturb.")

st.divider()

# Explain the "Bucket" Logic visually
st.subheader("How the Model Sees This Customer")
st.write("The model takes your inputs and sorts them into factors:")

# Simple bar chart of the 3 biggest factors
feature_impact = {
    "Contract Status": 0.4 if contract == "Month-to-month" else 0.1,
    "Price Sensitivity": monthly_charges / 120.0,
    "Loyalty (Tenure)": 1.0 - (tenure / 72.0) # Inverse: Low tenure = High Risk
}
st.bar_chart(feature_impact)
st.caption("Higher Bars = Higher Risk Contribution")