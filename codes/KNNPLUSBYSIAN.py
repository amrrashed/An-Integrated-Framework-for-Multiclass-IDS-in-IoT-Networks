import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from bayes_opt import BayesianOptimization

# Load your dataset
df = pd.read_csv('D:/new researches/Security/dataset/processed win dataset/windows10_dataset.csv')

# Extract features and labels
X = df.drop(columns=['label', 'type'])
y = df['type']

# Apply label encoding to y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check for string values in X and encode them if they exist
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        print(f"Encoded column: {column}")

# Split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=1)

# Load the encoder model from file
encoder = load_model('encoder1.h5')

# Encode the train data
X_train_encode = encoder.predict(X_train)

# Encode the test data
X_test_encode = encoder.predict(X_test)

# Combine train and test data for saving (if needed)
combined_data = np.concatenate((X_train_encode, X_test_encode), axis=0)

# Convert to DataFrame
combined_df = pd.DataFrame(combined_data, columns=[f"encoded_{i}" for i in range(X_train_encode.shape[1])])

# Optionally, you can save the encoded data to a CSV file
combined_df.to_csv("encoded_data.csv", index=False)

# Define the function to optimize
def knn_cv(n_neighbors, leaf_size, p):
    # Convert parameters to integers
    n_neighbors = int(n_neighbors)
    leaf_size = int(leaf_size)
    p = int(p)
    
    # Define the KNN model with given hyperparameters
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, p=p)
    
    # Fit the model on the training set
    knn_model.fit(X_train_encode, y_train)
    
    # Make predictions on the test set
    yhat = knn_model.predict(X_test_encode)
    
    # Calculate classification accuracy
    acc = accuracy_score(y_test, yhat)
    
    return acc

# Define the parameter space
pbounds = {
    'n_neighbors': (1, 10),
    'leaf_size': (10, 50),
    'p': (1, 2)
}

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=knn_cv,
    pbounds=pbounds,
    random_state=1
)

# Maximize the objective function (accuracy)
optimizer.maximize(init_points=10, n_iter=50)

# Get the best parameters
best_params = optimizer.max['params']
best_n_neighbors = int(best_params['n_neighbors'])
best_leaf_size = int(best_params['leaf_size'])
best_p = int(best_params['p'])

# Define the best KNN model with optimal hyperparameters
best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, leaf_size=best_leaf_size, p=best_p)

# Fit the model on the training set
best_knn_model.fit(X_train_encode, y_train)

# Make predictions on the test set
best_yhat = best_knn_model.predict(X_test_encode)

# Calculate classification accuracy
best_acc = accuracy_score(y_test, best_yhat)
print("Best Accuracy:", best_acc)
