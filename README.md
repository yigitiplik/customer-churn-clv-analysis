# customer-churn-clv-analysis
Customer churn prediction and CLV analysis using Python.

# Customer Churn Prediction & CLV Analysis

This project analyzes customer churn using the Telco Customer Churn dataset (original Kaggle dataset) and builds a predictive model supported by Customer Lifetime Value (CLV) segmentation.
The goal is to help businesses understand why customers leave and which customer groups carry the highest long-term value.

# Key Features
## Exploratory Data Analysis (EDA)

Missing value handling

Distribution checks

Churn vs non-churn comparisons

🤖 Machine Learning Model

Logistic Regression with:

OneHotEncoding

StandardScaler

ColumnTransformer Pipeline

ROC-AUC, classification report, confusion matrix

💸 Customer Lifetime Value (CLV)

Estimated CLV = MonthlyCharges × tenure

Segmentation: Low / Mid / High value customers

Churn rate by CLV segment

📈 Business Insights

Which customers churn the most?

Which contract types are risky?

Which CLV categories have the highest churn?

How retention strategies should change?

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-Learn

Pathlib

Jupyter / .py script

📁 Project Structure
customer-churn-clv-analysis/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── churn_clv_analysis.py
│
├── results/
│   └── (model outputs, charts will be added here)
│
└── README.md

▶️ How to Run
cd customer-churn-clv-analysis
python notebooks/churn_clv_analysis.py


This script performs:

Data loading

Cleaning & preprocessing

Model training

CLV estimation

Insight extraction (contract type, CLV segment, etc.)

Console output includes:

Classification report

ROC-AUC score

Confusion matrix

Churn rate by contract

Churn rate by CLV segment

💼 Business Value

This analysis helps organisations:

Identify customers who are most likely to leave

Understand high-risk contract types

Prioritize retention efforts based on CLV

Reduce churn-driven revenue loss

Improve long-term profitability

📌 Next Steps (Future Enhancements)

Add Random Forest / XGBoost models

Add SHAP explainability for feature importance

Build interactive dashboards (Streamlit / Power BI)

Calculate discounted CLV using probabilistic churn survival
