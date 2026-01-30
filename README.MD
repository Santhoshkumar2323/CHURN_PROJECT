Customer Churn Prediction System:

What this project is?

This is an end-to-end customer churn risk system that goes beyond prediction accuracy and focuses on decision usefulness.

It combines:

Data pipeline,
a preprocessing layer,
a supervised ML model,
business-optimized decision thresholds,
explainability and reporting,
an interactive Streamlit presentation.

What problem it solves

In real businesses, churn models fail when:
preprocessing breaks,
predictions lack explanation,
outputs don’t translate into actions
This project stress test those failure points.

System architecture:

The system is split into six clear stages:

1. Data ingestion:

Raw dataset is downloaded and saved, 
Train/Test split is done before any analysis,
Prevents data leakage.
This guarantees clean evaluation discipline.

2. Preprocessing pipeline:

A reusable preprocessing pipeline is built and serialized:
Fixes known dataset issues (TotalCharges text bug),
Separates numeric and categorical features,
Uses median imputation (for outliers),
Applies standard scaling and one-hot encoding,
Drops non-predictive identifiers.

This pipeline is:
trained once,
reused(training, inference, app),
immune to schema drift

3. Model training with business thresholding:

XGBoost classifier is trained on processed features.
Predictions are probabilities, not labels.
Optimal decision threshold is selected using F1 optimization, not accuracy.
Threshold is persisted and reused consistently.

4. Evaluation & metrics:

Model is evaluated on a held-out test set.
Metrics and feature names are stored.
No test data is ever seen during preprocessing or tuning

5. Explainability:

Two explainability layers exist:

A. SHAP analysis:

Generates global feature importance.
Translates encoded feature names into human language.
Produces a visual explanation report.

B. Clean business report:

Maps raw feature importance into interpretations.
Filters noise and highlights decision-relevant factors.
Outputs presentation-ready charts

A live dashboard allows:

simulation of customer profiles,
real-time churn probability scoring.
threshold-based risk classification,
concrete retention recommendations,

-This turns the model into an action system

Key design decisions:

Train/Test split before EDA to prevent leakage,
Serialized preprocessing pipeline to avoid runtime bugs,
Probability → threshold → decision separation,
Explainability as a first-class output.

Limitations:

Trained on a single public dataset,
Threshold optimization assumes symmetric business costs,
SHAP explanations are global, not causal,
Model performance will drift if customer behavior changes,
-These are acknowledged design limits.

How to run:

Build dataset:

python src/make_dataset.py

Build preprocessing pipeline:

python src/preprocessing.py

Train model:

python src/train_model.py

Run explainability:
python src/explainability.py
python src/simple_report.py

Launch app:

streamlit run src/app.py

-This is a complete, working churn decision system with clear separation between data, model, and business logic.


















