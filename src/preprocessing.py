import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# 1. CONFIGURATION
TRAIN_DATA_PATH = "data/processed/train.csv"
OUTPUT_PREPROCESSOR_PATH = "models/preprocessor.pkl"

# 2. DEFINE CUSTOM TRANSFORMERS
# This is "Durable" Engineering: We create a class that can fix the specific error
# in this dataset (strings instead of numbers) automatically.
class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Create a copy to avoid settingWithCopy warnings
        X_copy = X.copy()
        # Force convert to numeric, turning " " into NaN (Not a Number)
        X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
        return X_copy

# 3. MAIN EXECUTION
def run_preprocessing():
    print("⚙️  Building the Unbreakable Pipeline...")
    
    # Load Training Data
    df = pd.read_csv(TRAIN_DATA_PATH)
    
    # Separate Features (X) and Target (y)
    X = df.drop('Churn', axis=1)
    # We don't preprocess the target 'Churn' in the pipeline, we handle it in training
    
    # 4. DEFINE COLUMN GROUPS
    # We identify which columns are which. 
    # Note: 'customerID' is useless for prediction, so we will drop it.
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features and col != 'customerID']

    print(f"   Found {len(numeric_features)} numerical features.")
    print(f"   Found {len(categorical_features)} categorical features.")

    # 5. BUILD THE PIPELINES
    
    # Numeric Pipeline:
    # 1. Fix the "TotalCharges" text bug
    # 2. Fill missing values with the Median (robust to outliers)
    # 3. Scale numbers (StandardScaler) so big numbers don't dominate small ones
    numeric_transformer = Pipeline(steps=[
        ('cleaner', TotalChargesCleaner()), 
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Pipeline:
    # 1. Fill missing values with "missing"
    # 2. One-Hot Encode (Turn "Male/Female" into 0/1 columns)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine them into one Master Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop columns not specified (like customerID)
    )

    # 6. FIT AND SAVE
    # We "Learn" the medians and categories from the Training Data ONLY.
    print("   Fitting pipeline to training data...")
    preprocessor.fit(X)
    
    joblib.dump(preprocessor, OUTPUT_PREPROCESSOR_PATH)
    print(f"   ✅ Pipeline object saved to {OUTPUT_PREPROCESSOR_PATH}")
    print("   (This file now contains all the intelligence needed to clean new data)")

if __name__ == "__main__":
    run_preprocessing()