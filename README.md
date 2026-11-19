# FinRisk - Credit Card Fraud Detection System

## Overview

FinRisk is a machine learning-based credit card fraud detection project. It identifies potentially fraudulent transactions using historical transaction data and provides an interactive Streamlit dashboard for analysis.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## Dataset

Credit Card Fraud Detection Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Total transactions: 284,807
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Input features: 30
- Target: Class

The V1-V28 features are anonymized numerical variables.

## Methodology

1. Exploratory data analysis and class distribution analysis.
2. Feature selection and preprocessing.
3. Stratified training and testing split.
4. StandardScaler applied to Time and Amount.
5. Logistic Regression and Random Forest training.
6. Model comparison using classification metrics.
7. Random Forest threshold optimization using validation data.
8. Final evaluation on held-out test data.
9. Model serialization and transaction-level prediction.

## Model Performance

### Logistic Regression
- Precision: 6.10%
- Recall: 91.84%
- F1-score: 0.1144
- ROC-AUC: 0.9722

### Random Forest - Default Threshold
- Precision: 96.00%
- Recall: 73.47%
- F1-score: 0.8324
- ROC-AUC: 0.9529

### Threshold-Tuned Random Forest
- Threshold: 0.25
- Precision: 88.30%
- Recall: 84.69%
- F1-score: 0.8646
- ROC-AUC: 0.9529

Final test confusion matrix:

| | Predicted Legitimate | Predicted Fraudulent |
|---|---:|---:|
| Actual Legitimate | 56,853 | 11 |
| Actual Fraudulent | 15 | 83 |

## Dashboard Features

- CSV transaction upload
- Fraud probability prediction
- Transaction classification
- Risk summary and visualization
- Transaction filtering and review
- Downloadable prediction reports
- Model performance information

## Project Structure

    FinRisk/
    ├── .streamlit/
    │   └── config.toml
    ├── data/
    │   └── raw/
    ├── models/
    │   └── finrisk_random_forest.joblib
    ├── reports/
    │   ├── figures/
    │   ├── model_comparison.csv
    │   └── threshold_comparison.csv
    ├── app.py
    ├── main.py
    ├── predict.py
    ├── requirements.txt
    └── README.md

## Installation

Create and activate a virtual environment, then install dependencies:

    pip install -r requirements.txt

Place the downloaded dataset at:

    data/raw/creditcard.csv

## Running the Project

To train and evaluate the models:

    python main.py

To run the prediction demonstration:

    python predict.py

To launch the interactive dashboard:

    python -m streamlit run app.py

## Limitations

- V1-V28 are anonymized features.
- Predictions use the dataset's original feature representation.
- The application is an academic prototype and is not connected to a real payment gateway.
- Performance on other transaction datasets may differ.

## Future Improvements

- Additional model interpretability.
- Evaluation on newer transaction datasets.
- Credit-default prediction as a separate extension.
- Further dashboard improvements.

## Project Status

Core fraud detection pipeline implemented, evaluated and integrated with an interactive Streamlit dashboard.
