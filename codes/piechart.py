import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('D:/new researches/Security/dataset/train_test_network.csv')

# Extract features and labels
X = df.drop(columns=['label', 'type'])
y = df['type']

# Check for Missing Values in y
if y.isnull().sum() > 0:
    print(f"Warning: There are {y.isnull().sum()} missing values in the target variable.")
    df = df.dropna(subset=['type'])
    X = df.drop(columns=['label', 'type'])
    y = df['type']
else:
    print("No missing values in the target variable.")

# Plot class distribution pie chart
plt.figure(figsize=(10, 8), dpi=200)  # Increase figure size and DPI for high resolution
type_counts = y.value_counts()
type_labels = type_counts.index
type_sizes = type_counts.values

plt.pie(type_sizes, labels=type_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Class Distribution Pie Chart for train_test_network_dataset')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
