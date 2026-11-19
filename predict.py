
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Load the saved model bundle
bundle = joblib.load("models/finrisk_random_forest.joblib")

model = bundle["model"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]

# Load the dataset
df = pd.read_csv("data/raw/creditcard.csv")

# Separate features and actual labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Recreate the same train-test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Select one legitimate and one fraudulent test transaction
sample_indices = [
    y_test[y_test == 0].index[0],
    y_test[y_test == 1].index[0]
]

print("\n--- FINRISK TRANSACTION PREDICTIONS ---")
print("Decision threshold:", threshold)

for index in sample_indices:

    # Prepare the transaction
    transaction = X.loc[[index], feature_columns].copy()

    # Apply the saved scaler
    transaction[["Time", "Amount"]] = scaler.transform(
        transaction[["Time", "Amount"]]
    )

    # Predict fraud probability
    fraud_probability = model.predict_proba(transaction)[0][1]

    # Apply the selected threshold
    prediction = int(fraud_probability >= threshold)

    # Get actual class for demonstration
    actual = int(y.loc[index])

    actual_label = "Fraudulent" if actual == 1 else "Legitimate"
    predicted_label = "Fraudulent" if prediction == 1 else "Legitimate"

    print("\nTransaction ID:", index)
    print("Transaction Amount:", df.loc[index, "Amount"])
    print("Actual Class:", actual_label)
    print(f"Fraud Probability: {fraud_probability:.2%}")
    print("FinRisk Prediction:", predicted_label)

    if prediction == 1:
        print("Result: Transaction flagged for review.")
    else:
        print("Result: Transaction classified as legitimate.")
