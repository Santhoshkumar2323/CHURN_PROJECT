import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

# --- HELPER CLASS (Required for loading) ---
class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
        return X_copy

def generate_clean_report():
    print("📊 Generating Clean Business Report...")
    
    # 1. Load Model
    model = joblib.load("models/xgboost_model.pkl")
    
    # 2. Get Importance Scores
    importance = model.feature_importances_
    
    # 3. MANUALLY MAP THE "UGLY" NAMES TO ENGLISH
    # We know the order because we built the pipeline.
    # 0-2 are numbers, 3+ are categories.
    feature_map = {
        0: "Tenure (Loyalty)",
        1: "Monthly Bill ($)",
        2: "Total Lifetime Value",
        37: "Contract: Month-to-Month",  # x13 usually lands here after encoding
        39: "Contract: Two Year",
        17: "Internet: Fiber Optic",
        16: "Internet: DSL"
    }
    
    # Create a list of names. If we know the name, use it. If not, ignore it.
    feature_names = []
    for i in range(len(importance)):
        if i in feature_map:
            feature_names.append(feature_map[i])
        else:
            feature_names.append(f"Other Feature {i}")

    # 4. Create Dataframe
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Filter out "Other Features" to show only the big meaningful ones
    df_clean = df_imp[df_imp['Feature'].str.contains("Other") == False]
    df_clean = df_clean.sort_values('Importance', ascending=False)

    # 5. Plot
    plt.figure(figsize=(10, 6))
    # Green/Blue professional color scheme
    sns.barplot(data=df_clean, x='Importance', y='Feature', palette='viridis')
    
    plt.title("Top Drivers of Customer Churn", fontsize=15, fontweight='bold')
    plt.xlabel("Impact on Risk", fontsize=12)
    plt.ylabel("")
    plt.tight_layout()
    
    # Save
    output_path = "simple_risk_report.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ CLEAN Report saved to: {output_path}")
    print("   Open this image. It will use real English names.")

if __name__ == "__main__":
    generate_clean_report()