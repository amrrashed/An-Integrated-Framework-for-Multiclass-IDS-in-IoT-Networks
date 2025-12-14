import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.special import softmax

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def weighted_soft_voting(prob_preds, weights):
    weights = np.array(weights)
    weighted_sum = np.zeros_like(prob_preds[0])
    
    for i, probs in enumerate(prob_preds):
        weighted_sum += weights[i] * probs
    
    activated = softmax(weighted_sum, axis=1)
    return np.argmax(activated, axis=1)

def evaluate_weighted_voting(data_path, weights, num_features):
    df = pd.read_csv(data_path)
    
    X = df.drop(["label"], axis=1)
    y = df["label"]
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    selector = SelectKBest(score_func=f_classif, k=num_features)
    X_selected = selector.fit_transform(X, y_encoded)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Initialize classifiers
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_SEED)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        
        prob1 = rf.predict_proba(X_test)
        prob2 = xgb.predict_proba(X_test)
        prob3 = knn.predict_proba(X_test)
        
        prob_preds = [prob1, prob2, prob3]
        y_pred = weighted_soft_voting(prob_preds, weights)
        
        scores.append(accuracy_score(y_test, y_pred))
    
    return np.mean(scores)

# Example usage:
if __name__ == "__main__":
    data_path = r"D:/new researches/send/Security/BORUTA/win10_96featues.csv"
    
    weights = [0.4, 0.3, 0.3]  # Adjust as needed
    num_features = 70          # Choose optimal k based on prior analysis
    
    accuracy = evaluate_weighted_voting(data_path, weights, num_features)
    print(f"Mean Accuracy with RF, XGB, KNN (10-fold CV): {accuracy:.4f}")
