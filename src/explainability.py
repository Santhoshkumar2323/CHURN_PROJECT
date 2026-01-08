import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import sys
import os

# --- 1. SETUP & FIXES ---
# (We need the class definition again for the pickle to load)
from sklearn.base import BaseEstimator, TransformerMixin
class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
        return X_copy

# Configuration
TEST_DATA_PATH = "data/processed/test.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
MODEL_PATH = "models/xgboost_model.pkl"
REPORT_PATH = "churn_risk_report.png"

def make_features_readable(feature_names):
    """
    Translates machine names (x0, x1) into human names.
    Based on our pipeline structure:
    x0 = Tenure, x1 = MonthlyCharges, x2 = TotalCharges
    """
    feature_map = {
        "x0": "Tenure (Months)",
        "x1": "Monthly Charges ($)",
        "x2": "Total Charges ($)",
        "x13_Month-to-month": "Contract: Monthly",
        "x13_Two year": "Contract: 2-Year",
        "x13_One year": "Contract: 1-Year",
        "x17_Fiber optic": "Internet: Fiber Optic",
        "x17_DSL": "Internet: DSL"
    }
    
    clean_names = []
    for name in feature_names:
        # If we have a direct translation, use it
        if name in feature_map:
            clean_names.append(feature_map[name])
        # Otherwise, clean up the OneHot mess (e.g., "x3_Yes" -> "x3: Yes")
        else:
            clean_names.append(name.replace('_', ': '))
    return clean_names

def run_explanation():
    print("🕵️  Starting Model Investigator...")

    # 1. LOAD ARTIFACTS
    print("   Loading model and data...")
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        df_test = pd.read_csv(TEST_DATA_PATH)
    except Exception as e:
        print(f"   ❌ Error loading files: {e}")
        return

    # 2. PREPARE DATA (Transform raw test data into model format)
    X_test = df_test.drop('Churn', axis=1)
    # We use a small sample (100 rows) for speed, SHAP is slow on big data
    X_sample = X_test.sample(100, random_state=42)
    
    print("   Transforming data sample...")
    X_transformed = preprocessor.transform(X_sample)
    
    # 3. FIX NAMES
    # We assume standard feature ordering from the pipeline
    raw_feature_names = [f"x{i}" for i in range(X_transformed.shape[1])]
    
    # Try to get better names if possible, otherwise use the map
    try:
        # Attempt to pull names from OneHotEncoder
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out()
        raw_feature_names = ['x0', 'x1', 'x2'] + list(cat_names)
    except:
        pass # Fallback to x0, x1...

    human_feature_names = make_features_readable(raw_feature_names)

    # 4. CALCULATE SHAP (The "Why")
    print("   Calculating SHAP values (This explains the predictions)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # 5. GENERATE PLOT
    print(f"   Generating Summary Plot to {REPORT_PATH}...")
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_transformed, feature_names=human_feature_names, show=False)
    
    # Save nicely
    plt.tight_layout()
    plt.savefig(REPORT_PATH, bbox_inches='tight', dpi=300)
    print(f"   ✅ Report saved: {os.path.abspath(REPORT_PATH)}")
    
    # 6. TEXT EXPLANATION (For the user)
    print("\n   --- 📊 INSIGHTS GENERATED ---")
    print("   Open 'churn_risk_report.png' to see the drivers.")
    print("   HOW TO READ IT:")
    print("   1. RED dots = High value (e.g., High Price).")
    print("   2. BLUE dots = Low value (e.g., Low Tenure).")
    print("   3. If Red dots are on the RIGHT -> That feature INCREASES Churn.")
    print("   4. If Blue dots are on the RIGHT -> Low values INCREASE Churn.")

if __name__ == "__main__":
    run_explanation()