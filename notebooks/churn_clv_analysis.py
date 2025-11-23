"""
Churn & CLV Analysis
--------------------

Dataset: Telco Customer Churn (original Kaggle dataset)
File:    ../data/customer-churn-clv-analysisdatachurn.csv

This script performs:
- Basic data cleaning
- Exploratory checks
- Churn modelling with Logistic Regression
- Simple CLV estimation
- Churn & CLV segmentation for business insight
"""

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "customer-churn-clv-analysisdatachurn.csv"
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df[["customerID", "Churn"]].head())

# -------------------------------------------------------------------
# 2. Basic cleaning & feature engineering
# -------------------------------------------------------------------

# TotalCharges is sometimes stored as string with blanks -> convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows where TotalCharges could not be converted
df = df.dropna(subset=["TotalCharges"]).copy()

# Create target as 0/1
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

# Simple CLV estimation (very basic):
# CLV ≈ MonthlyCharges * tenure
# (In a real project, you might discount cash flows, consider churn prob, etc.)
df["EstimatedCLV"] = df["MonthlyCharges"] * df["tenure"]

# Create simple CLV segments
clv_q = df["EstimatedCLV"].quantile([0.33, 0.66]).values
low_thr, mid_thr = clv_q[0], clv_q[1]

def clv_segment(x):
    if x < low_thr:
        return "Low CLV"
    elif x < mid_thr:
        return "Mid CLV"
    else:
        return "High CLV"

df["CLV_Segment"] = df["EstimatedCLV"].apply(clv_segment)

print("\nChurn rate by CLV segment:")
print(
    df.groupby("CLV_Segment")["ChurnFlag"]
      .mean()
      .sort_values(ascending=False)
)

# -------------------------------------------------------------------
# 3. Train / Test split
# -------------------------------------------------------------------

# Drop ID and original target text column from features
feature_cols = df.columns.difference(["customerID", "Churn", "ChurnFlag"])
X = df[feature_cols].copy()
y = df["ChurnFlag"].copy()

# Identify numeric and categorical feature columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print("\nNumeric features:", numeric_cols)
print("Categorical features:", categorical_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------------------------------------------------
# 4. Preprocessing & model pipeline
# -------------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# A simple baseline Logistic Regression model
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

# -------------------------------------------------------------------
# 5. Train model
# -------------------------------------------------------------------

clf.fit(X_train, y_train)

# -------------------------------------------------------------------
# 6. Evaluation
# -------------------------------------------------------------------

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 3))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------------
# 7. Business insight: churn by contract & CLV
# -------------------------------------------------------------------

churn_contract = (
    df.groupby("Contract")["ChurnFlag"]
      .mean()
      .sort_values(ascending=False)
      .to_frame(name="ChurnRate")
)

print("\nChurn rate by Contract type:")
print(churn_contract)

churn_clv_seg = (
    df.groupby("CLV_Segment")["ChurnFlag"]
      .mean()
      .sort_values(ascending=False)
      .to_frame(name="ChurnRate")
)

print("\nChurn rate by CLV segment:")
print(churn_clv_seg)

print("\nScript finished successfully.")

