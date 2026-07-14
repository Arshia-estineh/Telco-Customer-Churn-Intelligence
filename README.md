# Telco Customer Churn Intelligence

An end-to-end machine learning and Streamlit analytics project for predicting telecom customer churn using the IBM Telco Customer Churn dataset. The project combines exploratory data analysis, production-ready preprocessing, class balancing with SMOTE, an MLP neural network classifier, and a modern executive dashboard for interactive churn-risk simulation.

## Project Overview

Customer churn is one of the most important retention challenges for subscription businesses. This project analyzes customer demographics, account information, services, contract type, billing behavior, and charges to estimate the probability that a customer will churn.

The solution is split into two main deliverables:

- `model_development.ipynb`: Jupyter notebook for data cleaning, EDA, preprocessing, model training, evaluation, and artifact export.
- `app.py`: Streamlit dashboard that loads the trained model artifacts and provides interactive churn prediction.

## Key Features

- Cleans and validates the `TotalCharges` field.
- Drops non-predictive customer identifiers.
- Applies label encoding to binary categorical features.
- Applies one-hot encoding to multi-class categorical features.
- Scales numerical features with `StandardScaler`.
- Handles class imbalance using SMOTE.
- Trains a Scikit-Learn `MLPClassifier` with three hidden layers: `64`, `32`, and `16` neurons.
- Uses early stopping to reduce overfitting.
- Reports precision, recall, F1-score, accuracy, and AUC-ROC.
- Saves model, scaler, encoders, preprocessor, target encoder, and feature schema with `joblib`.
- Provides a premium dark-themed Streamlit dashboard with business KPIs, EDA charts, customer input controls, and a churn probability gauge.

## Repository Structure

```text
.
├── app.py
├── model_development.ipynb
├── model_development.py
├── models/
│   ├── binary_label_encoder.joblib
│   ├── feature_schema.joblib
│   ├── mlp_churn_model.joblib
│   ├── one_hot_encoder.joblib
│   ├── preprocessor.joblib
│   ├── scaler.joblib
│   └── target_encoder.joblib
└── README.md
```

## Dataset

This project uses the Telco Customer Churn dataset:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains customer-level records with features such as tenure, contract type, internet service, monthly charges, total charges, and churn status.

If you publish this project on GitHub, place the CSV in the project folder or update the dataset path in both `model_development.ipynb` and `app.py`.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn plotly streamlit joblib
```

## Model Development Workflow

Open and run the notebook:

```bash
jupyter notebook model_development.ipynb
```

The notebook performs:

1. Data loading and cleaning.
2. Exploratory data analysis.
3. Feature encoding and scaling.
4. SMOTE class balancing.
5. MLP model training.
6. Model evaluation.
7. Artifact export into the `models/` folder.

## Running the Streamlit App

After running the notebook and generating the model artifacts, start the dashboard:

```bash
streamlit run app.py
```

The dashboard includes:

- Total customer count.
- Overall churn rate.
- Model accuracy.
- Churn distribution chart.
- Churn rate by contract type.
- Monthly charges vs churn box plot.
- Interactive customer scenario inputs.
- Churn probability gauge.

## Model Summary

The model uses a Scikit-Learn MLP neural network with the following architecture:

```text
Input features -> Dense(64, relu) -> Dense(32, relu) -> Dense(16, relu) -> Churn probability
```

The trained model is designed for churn-risk prioritization and business decision support. It should be validated with fresh data before use in production.

## Business Use Cases

- Identify high-risk customers for retention campaigns.
- Compare churn risk across contract and billing segments.
- Support customer success teams with proactive outreach.
- Simulate churn probability for hypothetical customer profiles.
- Communicate churn drivers through an executive-friendly dashboard.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn
- Plotly
- Streamlit
- Joblib

## Notes

- The Streamlit app expects the trained artifacts to exist in the `models/` folder.
- If artifacts are missing, run `model_development.ipynb` first.
- The app preprocesses user input with the same saved preprocessing pipeline used during training.
- This project is intended as a professional portfolio project and can be extended with model monitoring, SHAP explanations, batch scoring, and deployment to Streamlit Community Cloud.

## License

This project is provided for educational and portfolio use. Add your preferred license before publishing publicly.
