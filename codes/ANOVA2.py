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
import missingno as msno

# Load dataset
df = pd.read_csv(
    'D:/new researches/send/Security/dataset/train_test_network.csv',
    sep=',',
    low_memory=False
)

# --- Data Summary: Data Types and Missing Values ---
print("\n--- Dataset Summary ---")
print("Column Data Types:")
print(df.dtypes)

columns_to_drop = [
    'src_ip', 'dst_ip', 'dns_query', 'ssl_subject',
    'ssl_issuer', 'http_user_agent'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')


# Replace dash with NaN
df.replace('-', np.nan, inplace=True)

# Convert appropriate columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Impute numerical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Impute categorical columns
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
    elif df[col].dtype == 'int64':
        df[col] = df[col].astype('int32')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(['label', 'type'], axis=1)),
                         columns=df.drop(['label', 'type'], axis=1).columns)

X = df.drop(columns=['label', 'type'])
y = df['type']  # or df['label'], depending on your classification task

# Encode target
y = y.astype(str)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Data Summary: Data Types and Missing Values ---
print("\n--- Dataset Summary ---")
print("Column Data Types:")
print(df.dtypes)

print(X.shape)
print(y.shape)

# Check for missing values and calculate percentages
missing_info = df.isnull().sum()
missing_percent = (missing_info / len(df)) * 100

print("\n--- Missing Values Report ---")
print(pd.DataFrame({
    'Missing Values': missing_info,
    'Percentage (%)': missing_percent,
    'Data Type': df.dtypes
})) 

# Class distribution
label_counts = df['type'].value_counts()
print("\nNumber of instances per label in 'type' column:")
print(label_counts)

# Plot class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=df)
plt.title('Class Distribution Histogram')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# ANOVA Feature Ranking
F_values, p_values = f_classif(X, y_encoded)
feature_scores = pd.DataFrame({"Feature": X.columns, "F_value": F_values})
feature_scores = feature_scores.sort_values(by="F_value", ascending=False)

# 10-Fold Cross Validation Setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = []
feature_range = list(range(5, 11, 2))  # Try top-k features from 5 to 40

# KNN Evaluation across feature ranges
for k in tqdm(feature_range):
    top_k_features = feature_scores.iloc[:k]['Feature'].values
    X_k = X[top_k_features].values
    knn_acc_list = []

    for train_index, test_index in kf.split(X_k):
        X_train, X_test = X_k[train_index], X_k[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # KNN classifier
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        knn_acc_list.append(accuracy_score(y_test, predictions))

    # Store average accuracy
    results.append({
        'num_features': k,
        'knn_accuracy': np.mean(knn_acc_list)
    })

# Results as DataFrame
results_df = pd.DataFrame(results)

# Find best performance
max_idx = results_df['knn_accuracy'].idxmax()
max_k = results_df.loc[max_idx, 'num_features']
max_acc = results_df.loc[max_idx, 'knn_accuracy']

# Plot Accuracy vs Feature Count
plt.figure(figsize=(12, 6))
plt.plot(results_df['num_features'], results_df['knn_accuracy'], label='KNN Accuracy', marker='o')
plt.scatter(max_k, max_acc, color='red', label=f'Max Accuracy: {max_acc:.4f} at {max_k} features', zorder=5)
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy (10-Fold CV)')
plt.title('KNN Accuracy vs. Number of Selected Features (ANOVA + K-Fold CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final result
print(f"\nMaximum KNN accuracy of {max_acc:.4f} achieved with {max_k} features.")

