# Import necessary libraries
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Load Dataset ---
data_path = 'D:/new researches/send/Security/dataset/processed win dataset/windows10_dataset.csv'
df = pd.read_csv(data_path, sep=',', low_memory=False)

# Handle missing values by filling them with the mean of numeric columns
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

# Validate label column existence
label_col = 'type'
if label_col not in df.columns:
    raise ValueError(f"'{label_col}' not found in dataset")

# Extract features and labels
X = df.drop(columns=['label', 'type'], errors='ignore')
y = df[label_col]

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Encode any string columns in features
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        print(f"Encoded column: {column}")

# Function to create shadow features
def add_shadow_features(X, seed):
    # Create a copy of X to avoid modifying the original dataframe
    X_shadow = X.copy()

    # Create a list to store all shadow columns
    shadow_columns = []

    for col in X.columns:
        # Shuffle each column to create a shadow feature
        shadow_columns.append(X[col].sample(frac=1, random_state=seed).reset_index(drop=True))

    # Concatenate all shadow columns at once
    X_shadow = pd.concat([X_shadow] + [pd.Series(shadow, name=f"shadow_{col}") for col, shadow in zip(X.columns, shadow_columns)], axis=1)
    
    return X_shadow

# Function to get important features using Boruta logic
def get_important_features(X_shadow, y):
    rf = RandomForestClassifier(max_depth=20, random_state=42)
    rf.fit(X_shadow, y)
    importances = dict(zip(X_shadow.columns, rf.feature_importances_))
    shadow_importances = {k: v for k, v in importances.items() if 'shadow_' in k}
    highest_shadow = max(shadow_importances.values())
    selected = [k for k, v in importances.items() if 'shadow_' not in k and v > highest_shadow]
    return selected

# Perform Boruta selection over multiple trials
TRIALS = 50
feature_hits = {f: 0 for f in X.columns}
important_features_df = pd.DataFrame(columns=['trial', 'important_features'])

for trial in tqdm(range(TRIALS)):
    X_shadow = add_shadow_features(X, seed=trial)
    imp_features = get_important_features(X_shadow, y_encoded)
    important_features_df = pd.concat([
        important_features_df,
        pd.DataFrame({'trial': [trial], 'important_features': [imp_features]})
    ], ignore_index=True)
    for feat in imp_features:
        if feat in feature_hits:
            feature_hits[feat] += 1

# Save feature hit counts
print("Feature hit counts:", feature_hits)
important_features_df.to_csv("important_features_with_labels.csv", index=False)

# Calculate binomial PMF
pmf = [sp.stats.binom.pmf(x, TRIALS, 0.5) for x in range(TRIALS + 1)]

def get_tail_items(pmf):
    total = 0
    for i, p in enumerate(reversed(pmf)):
        total += p
        if total >= 0.05:
            return len(pmf) - 1 - i
    return len(pmf) - 1

# Plot PMF
plt.plot(range(TRIALS + 1), pmf, "-o")
plt.title(f"Binomial distribution for {TRIALS} trials")
plt.xlabel("No. of trials")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# Choose features based on hit count thresholds
def choose_features(feature_hits, TRIALS, thresh):
    green_thresh = TRIALS - thresh
    blue_upper = green_thresh
    blue_lower = thresh
    green = [k for k, v in feature_hits.items() if v >= green_thresh]
    blue = [k for k, v in feature_hits.items() if blue_lower <= v < blue_upper]
    return green, blue

# Final feature selection
thresh = get_tail_items(pmf)
green, blue = choose_features(feature_hits, TRIALS, thresh)
print("Green zone features:", green)
print("Blue zone features:", blue)

# Create new dataframe with selected features
green_df = df[green + [label_col]].copy()
green_df.fillna(green_df.select_dtypes(include=[np.number]).mean(), inplace=True)

for column in green_df.columns:
    if green_df[column].dtype == 'object':
        le = LabelEncoder()
        green_df[column] = le.fit_transform(green_df[column].astype(str))
        print(f"Encoded column: {column}")

# Save final selected features
green_df.to_csv("selected_features_with_labels.csv", index=False)
