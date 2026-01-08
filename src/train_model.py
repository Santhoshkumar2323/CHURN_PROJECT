import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.base import BaseEstimator, TransformerMixin
import json
import sys
import os

# 1. CONFIGURATION
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
# CHANGED: We use .pkl now because we are using joblib (Standard Python format)
MODEL_PATH = "models/xgboost_model.pkl" 
METRICS_PATH = "models/metrics.json"

# --- THE CLASS DEFINITION --
# Required for pickle to know what 'TotalChargesCleaner' is
class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
        return X_copy
# ---------------------------

def get_profit_optimized_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def train():
    print("🧠 Starting Model Training...")
    
    # 2. LOAD DATA
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # 3. LOAD PREPROCESSOR
    print("   Loading the Unbreakable Pipeline...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 4. TRANSFORM DATA
    print("   Transforming data...")
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    try:
        num_cols = preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out()
        cat_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
        feature_names = list(num_cols) + list(cat_cols)
    except:
        feature_names = [f"feat_{i}" for i in range(X_train_processed.shape[1])]

    # 5. TRAIN XGBOOST
    print("   Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(X_train_processed, y_train)
    
    # 6. EVALUATE & TUNE
    print("   Calculating Optimal Business Threshold...")
    y_prob_train = model.predict_proba(X_train_processed)[:, 1]
    best_threshold, best_f1 = get_profit_optimized_threshold(y_train, y_prob_train)
    
    y_prob_test = model.predict_proba(X_test_processed)[:, 1]
    y_pred_test = (y_prob_test >= best_threshold).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"   ✅ Training Complete.")
    print(f"      Optimal Threshold: {best_threshold:.4f}")
    print(f"      Test Set Accuracy: {test_accuracy:.4f}")
    
    # 7. SAVE ARTIFACTS (FIXED)
    # We use joblib instead of model.save_model to avoid the bug
    joblib.dump(model, MODEL_PATH)
    
    metrics = {
        "threshold": float(best_threshold),
        "test_accuracy": float(test_accuracy),
        "feature_names": list(feature_names)
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
        
    print(f"   ✅ Model saved to {MODEL_PATH}")
    print(f"   ✅ Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    train()