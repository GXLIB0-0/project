# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Step 1: Load the dataset

# If you're using Google Colab, you can upload the dataset
try:
    from google.colab import files
    uploaded = files.upload()  # Upload the file using this method
    data = pd.read_csv('creditcard.csv')  # Assuming the dataset is uploaded with this name
except ImportError:
    # If running locally, ensure the file is in the same directory or specify the full path
    data = pd.read_csv('creditcard.csv')  # Replace with the actual path to your CSV file

# Step 2: Data Preprocessing
# Check for missing values
print(f"Missing values:\n{data.isnull().sum()}")

# Inspect the dataset
print(f"\nDataset Info:\n{data.info()}")
print(f"\nFirst few rows of the dataset:\n{data.head()}")

# Step 3: Exploratory Data Analysis (EDA)
# Check the distribution of fraud (Class column)
sns.countplot(x='Class', data=data)
plt.title("Class Distribution: 0=Not Fraud, 1=Fraud")
plt.show()

# Step 4: Split the data into features and target
X = data.drop(columns=['Class'])  # Features (dropping the target column)
y = data['Class']  # Target (fraud or not)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 6: Feature scaling (important for models like logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Handling imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Step 8: Train the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Step 9: Evaluate the model
y_pred = model.predict(X_test_scaled)

# Print classification report
print(f"\nRandom Forest Classification Report:\n{classification_report(y_test, y_pred)}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Step 10: Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy * 100:.2f}%')

# Model Comparison with Logistic Regression
# Logistic Regression can also be used for comparison
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_res, y_train_res)

# Evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test_scaled)
print(f"\nLogistic Regression Classification Report:\n{classification_report(y_test, y_pred_lr)}")

# Confusion matrix for Logistic Regression
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Accuracy for Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%')
