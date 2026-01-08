Customer Churn Risk System

A clean, end-to-end churn modeling pipeline built on the public IBM Telco Customer Churn dataset.

It ingests real customer data, applies a durable preprocessing pipeline, trains an XGBoost risk model, optimizes a business-driven decision threshold, explains predictions using SHAP, and serves decisions through an interactive Streamlit app.

Use case: Detect high-risk customers early and trigger retention actions.

It includes:

clean train/test data separation

reusable preprocessing pipeline

XGBoost classification model

probability threshold tuned for decision-making

model explainability using SHAP

Streamlit app for interactive risk inspection

Stack: Python · scikit-learn · XGBoost · SHAP · Streamlit