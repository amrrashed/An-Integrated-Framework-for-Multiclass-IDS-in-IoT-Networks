# Optimized Script with 10-Fold Cross-Validation and ANOVA-based Feature Ranking (KNN Only)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

#"D:\new researches\send\Security\dataset\train_test_network.csv"
#D:/new researches/send/Security/dataset/processed win dataset/windows10_dataset.csv
# Load dataset with improved type handling
df = pd.read_csv(
    'D:/new researches/send/Security/dataset/processed win dataset/windows10_dataset.csv',
    sep=',',
    low_memory=False
)
#df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Separate features and target
X = df.drop(columns=['label', 'type'])
y = df['type']

# Encode target
y = y.astype(str)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Encode categorical features if any
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = X[column].astype('category').cat.codes

# Count instances per label in 'type' column
label_counts = df['type'].value_counts()
print("Number of instances per label in 'type' column:")
print(label_counts)

# Plot class distribution histogram using 'type' column
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='type', data=df)
plt.title('Class Distribution Histogram')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Save the encoded dataset
#encoded_df = X.copy()
#encoded_df['type'] = y_encoded
#encoded_df.to_csv('encoded_windows10_dataset.csv', index=False)

# Apply ANOVA F-test for feature ranking
F_values, p_values = f_classif(X, y_encoded)
feature_scores = pd.DataFrame({"Feature": X.columns, "F_value": F_values})
feature_scores = feature_scores.sort_values(by="F_value", ascending=False)

# K-Fold Cross Validation (10 folds)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize result storage
results = []
feature_range = list(range(5, 42, 2))

# Evaluate KNN over range of top features
for k in tqdm(feature_range):
    top_k_features = feature_scores.iloc[:k]['Feature'].values
    X_k = X[top_k_features].values

    knn_acc_list = []

    for train_index, test_index in kf.split(X_k):
        X_train, X_test = X_k[train_index], X_k[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # KNN
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_acc_list.append(accuracy_score(y_test, knn.predict(X_test)))

    # Store average results
    results.append({
        'num_features': k,
        'knn_accuracy': np.mean(knn_acc_list)
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Find max accuracy point
max_idx = results_df['knn_accuracy'].idxmax()
max_k = results_df.loc[max_idx, 'num_features']
max_acc = results_df.loc[max_idx, 'knn_accuracy']

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(results_df['num_features'], results_df['knn_accuracy'], label='KNN', marker='o')
plt.scatter(max_k, max_acc, color='red', label=f'Max Accuracy: {max_acc:.4f} at {max_k} features', zorder=5)
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy (10-Fold CV)')
plt.title('KNN Accuracy vs. Number of Selected Features (ANOVA + K-Fold CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print number of features with the maximum accuracy
print(f"Maximum KNN accuracy of {max_acc:.4f} achieved with {max_k} features.")