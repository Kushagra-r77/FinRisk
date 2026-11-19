import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset
data_path = "data/raw/creditcard.csv"
df = pd.read_csv(data_path)

# Display basic information
print("\n--- First 5 Transactions ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Missing Values ---")
print(df.isnull().sum().sum())

print("\n--- Transaction Class Counts ---")
print(df["Class"].value_counts())

print("\n--- Class Percentages ---")
print((df["Class"].value_counts(normalize=True) * 100).round(4))


import matplotlib.pyplot as plt

# Count legitimate and fraudulent transactions
class_counts = df["Class"].value_counts().sort_index()

labels = ["Legitimate", "Fraudulent"]
counts = [class_counts.loc[0], class_counts.loc[1]]

# Create the bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, counts)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

# Display the exact count above each bar
plt.bar_label(bars, fmt="%.0f", padding=3)

plt.tight_layout()

# Save the chart
plt.savefig(
    "reports/figures/class_distribution.png",
    dpi=150
)

print("\nChart saved to reports/figures/class_distribution.png")

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create copies to avoid modifying the original data
X_train = X_train.copy()
X_test = X_test.copy()

# Scale Time and Amount
scaler = StandardScaler()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# Display the results
print("\n--- Data Preprocessing Results ---")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nFraud cases in training data:")
print(y_train.value_counts())

print("\nFraud cases in testing data:")
print(y_test.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

# Create the Logistic Regression model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the model
print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\n--- Logistic Regression Results ---")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Additional evaluation metrics
print("\nROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 4))
print("Average Precision:", round(average_precision_score(y_test, y_prob), 4))

# Visualize the confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraudulent"]
)

fig, ax = plt.subplots(figsize=(6, 5))
display.plot(ax=ax, values_format="d", colorbar=False)

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/figures/logistic_regression_confusion_matrix.png", dpi=150)
plt.show()

print("\nConfusion matrix graph saved successfully.")

# ---------------------------------
# RANDOM FOREST MODEL
# ---------------------------------

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# Train the model
print("\nTraining Random Forest...")
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\n--- Random Forest Results ---")

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_pred,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

# Confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix:")
print(rf_cm)

# Additional metrics
print("\nROC-AUC Score:", round(roc_auc_score(y_test, rf_prob), 4))
print("Average Precision:", round(average_precision_score(y_test, rf_prob), 4))

# Plot confusion matrix
rf_display = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["Legitimate", "Fraudulent"]
)

fig, ax = plt.subplots(figsize=(6, 5))
rf_display.plot(ax=ax, values_format="d", colorbar=False)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/figures/random_forest_confusion_matrix.png",
    dpi=150
)

plt.show()

print("\nRandom Forest confusion matrix saved successfully.")

# ---------------------------------
# MODEL COMPARISON
# ---------------------------------

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],

    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, rf_pred)
    ],

    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, rf_pred)
    ],

    "F1-Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, rf_pred)
    ],

    "ROC-AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, rf_prob)
    ],

    "Average Precision": [
        average_precision_score(y_test, y_prob),
        average_precision_score(y_test, rf_prob)
    ]
})

print("\n--- MODEL COMPARISON ---")
print(comparison.round(4).to_string(index=False))

# Save comparison results
comparison.to_csv("reports/model_comparison.csv", index=False)

# Visualize key evaluation metrics
metrics = comparison.set_index("Model")[
    ["Precision", "Recall", "F1-Score"]
]

ax = metrics.plot(
    kind="bar",
    figsize=(9, 5),
    ylim=(0, 1),
    rot=0
)

ax.set_title("Logistic Regression vs Random Forest")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig("reports/figures/model_comparison.png", dpi=150)
plt.show()

print("\nModel comparison saved successfully.")

# ---------------------------------
# RANDOM FOREST THRESHOLD TUNING
# ---------------------------------

from sklearn.model_selection import train_test_split

# Split training data into model-training and validation sets
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# Train a separate Random Forest for threshold experiments
rf_tuning_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest for threshold tuning...")
rf_tuning_model.fit(X_fit, y_fit)

# Get fraud probabilities for validation data
val_prob = rf_tuning_model.predict_proba(X_val)[:, 1]

# Experiment with different classification thresholds
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50]

threshold_results = []

for threshold in thresholds:

    # Convert probabilities into class predictions
    val_pred = (val_prob >= threshold).astype(int)

    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_pred,
        labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, val_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, val_pred, zero_division=0
        ),
        "F1-Score": f1_score(
            y_val, val_pred, zero_division=0
        ),
        "False Positives": fp,
        "False Negatives": fn
    })

# Display the results
threshold_df = pd.DataFrame(threshold_results)

print("\n--- RANDOM FOREST THRESHOLD COMPARISON ---")
print(threshold_df.round(4).to_string(index=False))

# Save the experiment results
threshold_df.to_csv(
    "reports/threshold_comparison.csv",
    index=False
)

print("\nThreshold comparison saved successfully.")

# ---------------------------------
# FINAL THRESHOLD-TUNED EVALUATION
# ---------------------------------

import joblib

# Select threshold based on validation results
selected_threshold = 0.25

# Apply the threshold to the original test predictions
tuned_pred = (rf_prob >= selected_threshold).astype(int)

print("\n--- FINAL RANDOM FOREST EVALUATION ---")
print("Selected Threshold:", selected_threshold)

print("\nClassification Report:")
print(classification_report(
    y_test,
    tuned_pred,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

# Calculate confusion matrix
tuned_cm = confusion_matrix(y_test, tuned_pred)

print("\nConfusion Matrix:")
print(tuned_cm)

tn, fp, fn, tp = tuned_cm.ravel()

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\nROC-AUC:", round(roc_auc_score(y_test, rf_prob), 4))
print(
    "Average Precision:",
    round(average_precision_score(y_test, rf_prob), 4)
)

# Plot the final confusion matrix
tuned_display = ConfusionMatrixDisplay(
    confusion_matrix=tuned_cm,
    display_labels=["Legitimate", "Fraudulent"]
)

fig, ax = plt.subplots(figsize=(6, 5))
tuned_display.plot(ax=ax, values_format="d", colorbar=False)

plt.title("Threshold-Tuned Random Forest (0.25)")
plt.tight_layout()

plt.savefig(
    "reports/figures/tuned_random_forest_confusion_matrix.png",
    dpi=150
)

plt.show()

# Save model and preprocessing information together
joblib.dump(
    {
        "model": rf_model,
        "scaler": scaler,
        "threshold": selected_threshold,
        "feature_columns": list(X_train.columns)
    },
    "models/finrisk_random_forest.joblib"
)

print("\nFinal model saved to models/finrisk_random_forest.joblib")
