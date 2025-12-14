# Optimized Script with 10-Fold Cross-Validation and ANOVA-based Feature Ranking (RF Only)

# --- Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import missingno as msno
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- Load Dataset ---
df = pd.read_csv(
    'D:/new researches/send/Security/dataset/train_test_network.csv',
    sep=',',
    low_memory=False
)

# --- Drop High Cardinality or Redundant Columns ---
columns_to_drop = [
    'src_ip', 'dst_ip', 'dns_query', 'ssl_subject',
    'ssl_issuer', 'http_user_agent'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# --- Replace '-' with NaN for Standardized Missing Value Handling ---
df.replace('-', np.nan, inplace=True)

# --- Attempt to Convert All Columns to Numeric Where Applicable ---
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# --- Impute Numerical Columns with Mean ---
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# --- Impute Categorical Columns with Mode ---
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# --- Label Encode All Categorical Columns ---
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    

# --- Downcast Data Types to Optimize Memory Usage ---
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
    elif df[col].dtype == 'int64':
        df[col] = df[col].astype('int32')

# --- Feature Scaling (Not used in RandomForest but good for consistency) ---
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df.drop(['label', 'type'], axis=1)),
    columns=df.drop(['label', 'type'], axis=1).columns
)

# --- Split Features and Labels ---
X = df.drop(columns=['label', 'type'])  # Features
y = df['type']  # Target label (could use 'label' depending on your problem)

# --- Encode Target Variable ---
y = y.astype(str)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Show Dataset Structure ---
print("\n--- Dataset Summary ---")
print("Column Data Types:")
print(df.dtypes)
print("\nShape of Features and Target:")
print(X.shape)
print(y.shape)

# --- Missing Value Report ---
missing_info = df.isnull().sum()
missing_percent = (missing_info / len(df)) * 100
print("\n--- Missing Values Report ---")
print(pd.DataFrame({
    'Missing Values': missing_info,
    'Percentage (%)': missing_percent,
    'Data Type': df.dtypes
}))

# --- Class Distribution ---
label_counts = df['type'].value_counts()
print("\nNumber of instances per label in 'type' column:")
print(label_counts)

# --- Plot Class Distribution ---
plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=df)
plt.title('Class Distribution Histogram')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# --- ANOVA F-test for Feature Selection ---
F_values, p_values = f_classif(X, y_encoded)
feature_scores = pd.DataFrame({"Feature": X.columns, "F_value": F_values})
feature_scores = feature_scores.sort_values(by="F_value", ascending=False)

# --- 10-Fold Cross Validation Setup ---
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = []
feature_range = list(range(5, 35, 2))  # Evaluate top-k features from 5 to 34

# --- Evaluate Random Forest Classifier for Varying Number of Top Features ---
for k in tqdm(feature_range):
    top_k_features = feature_scores.iloc[:k]['Feature'].values
    X_k = X[top_k_features].values
    rf_acc_list = []

    # Perform K-Fold Cross-Validation
    for train_index, test_index in kf.split(X_k):
        X_train, X_test = X_k[train_index], X_k[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Initialize and Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        rf_acc_list.append(accuracy_score(y_test, predictions))

    # Store mean accuracy for current feature count
    results.append({
        'num_features': k,
        'rf_accuracy': np.mean(rf_acc_list)
    })

# --- Convert Results to DataFrame ---
results_df = pd.DataFrame(results)

# --- Find the Best Performing Feature Count ---
max_idx = results_df['rf_accuracy'].idxmax()
max_k = results_df.loc[max_idx, 'num_features']
max_acc = results_df.loc[max_idx, 'rf_accuracy']

# --- Plot Accuracy vs Number of Selected Features ---
plt.figure(figsize=(12, 6))
plt.plot(results_df['num_features'], results_df['rf_accuracy'], label='Random Forest Accuracy', marker='o')
plt.scatter(max_k, max_acc, color='red', label=f'Max Accuracy: {max_acc:.4f} at {max_k} features', zorder=5)
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy (10-Fold CV)')
plt.title('Random Forest Accuracy vs. Number of Selected Features (ANOVA + K-Fold CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Output Best Result ---
print(f"\nMaximum Random Forest accuracy of {max_acc:.4f} achieved with {max_k} features.")
