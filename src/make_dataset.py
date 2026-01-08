import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 1. CONSTANTS (Configuration)
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_PATH = "data/raw/telco_churn.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

def make_dataset():
    print("🚀 Starting Data Ingestion...")
    
    # 2. Download Data
    print(f"   Downloading from {URL}...")
    df = pd.read_csv(URL)
    
    # 3. Save Raw Data (The "Backup")
    # We create the directory if it doesn't exist
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"   ✅ Saved raw data to {RAW_PATH} ({df.shape})")

    # 4. The "Holy" Split
    # We split NOW, before we do any cleaning or analysis.
    # This guarantees we never accidentally see the test data.
    print("   Splitting data into Train/Test sets...")
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])
    
    # 5. Save Processed Splits
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    
    print(f"   ✅ Saved Train set to {TRAIN_PATH} ({train.shape})")
    print(f"   ✅ Saved Test set to {TEST_PATH} ({test.shape})")
    print("\n🎉 Data pipeline complete. You may now begin EDA on 'data/processed/train.csv'.")

if __name__ == "__main__":
    make_dataset()