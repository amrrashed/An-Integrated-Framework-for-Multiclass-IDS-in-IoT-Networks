import pandas as pd
import numpy as np
import random
from sklearn.ensemble import ExtraTreesClassifier
from boruta import BorutaPy

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Load Dataset ---
data_path = 'D:/new researches/send/Security/dataset/train_test_network.csv'
df = pd.read_csv(data_path, sep=',', low_memory=False)

# Handle missing values by filling them with the mean of numeric columns
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

# Define features and target
label_col = 'type'
X = df.drop(columns=[label_col])
y = df[label_col]

# Encode categorical columns if any
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.factorize(X[col].astype(str))[0]

# Initialize ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, class_weight='balanced', random_state=42)

# Initialize Boruta
boruta_selector = BorutaPy(estimator=et, n_estimators='auto', random_state=42)

# Fit Boruta
boruta_selector.fit(X.values, y.values)

# Get selected and tentative features
selected_features = X.columns[boruta_selector.support_]
tentative_features = X.columns[boruta_selector.support_weak_]

# Print results
print("Selected Features:")
print(selected_features.tolist())
print("Tentative Features:")
print(tentative_features.tolist())

# Reduced dataset
X_selected = X[selected_features].copy()
X_selected["label"] = y.values

# Save results
output_path = r"selected_features_with_labels_train_test_network.csv"
X_selected.to_csv(output_path, index=False)
print(f"Reduced dataset with labels saved to {output_path}")
