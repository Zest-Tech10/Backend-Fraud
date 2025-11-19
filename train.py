# backend/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

print("Starting model training...")

# 1. Load Data
try:
    # data = pd.read_csv('creditcard.csv')
    DATA_URL = "https://drive.google.com/uc?export=download&id=1jBGMEyO_lfwwFxDAQkQEu_nXJj1LSTa9"
    data = pd.read_csv(DATA_URL)
except FileNotFoundError:
    print("ERROR: creditcard.csv not found.")
    print("Please download it from Kaggle and place it in the 'backend' folder.")
    exit()

# 2. Prepare Data
# আমরা ডেটাকে ব্যালান্স করার জন্য Undersampling করবো
normal = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

# ফ্রডের সংখ্যার সমান নরমাল ডেটা নেওয়া
normal_sample = normal.sample(n=len(fraud), random_state=42)
data = pd.concat([normal_sample, fraud], axis=0)

# 3. Feature Scaling
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
# Time কলামটি আমরা ড্রপ করে দিচ্ছি
data = data.drop(['Time'], axis=1)

# 4. Define X and y
X = data.drop(['Class'], axis=1)
y = data['Class']

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. Save Model and Scaler
joblib.dump(model, 'fraud_model.joblib')
joblib.dump(scaler, 'scaler.joblib') # Amount স্কেলারও সেভ করছি

print("Model training complete!")
print("Files created: 'fraud_model.joblib', 'scaler.joblib'")