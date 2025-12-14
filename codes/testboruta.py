# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Read the CSV file containing the selected features with labels
df = pd.read_csv("selected_features_with_labels.csv")

# Encode string features if any exist
for column in df.columns:
    if df[column].dtype == 'object' and column != 'type':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        print(f"Encoded column: {column}")

# Split the selected features into features and target variable
X_selected = df.drop(columns=['type'])
y_selected = LabelEncoder().fit_transform(df['type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.2, random_state=1)

# Train a Random Forest model on the selected features
rf_model = RandomForestClassifier(max_depth=20)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with selected features:", accuracy)

# Calculate AUC and plot ROC curve for multi-class classification
n_classes = len(np.unique(y_selected))
y_test_binarized = label_binarize(y_test, classes=range(n_classes))

# Compute AUC for each class
auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')
print("AUC:", auc)

# Compute ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])

# Plot all ROC curves
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-class')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Calculate specificity for each class
specificity = []
for i in range(n_classes):
    tn = conf_matrix[i, i]
    fp = conf_matrix[:, i].sum() - tn
    fn = conf_matrix[i, :].sum() - tn
    tp = conf_matrix.sum() - (tn + fp + fn)
    specificity.append(tn / (tn + fp))
print("Specificity:", specificity)

# Generate classification report for F-score, precision, recall
class_report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(n_classes)])
print("Classification Report:\n", class_report)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, [str(i) for i in range(n_classes)], rotation=45)
plt.yticks(tick_marks, [str(i) for i in range(n_classes)])

# Normalize the confusion matrix
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, f"{conf_matrix[i, j]:.2f}", horizontalalignment="center", color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
