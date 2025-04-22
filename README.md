# AI MSE Submission: Health Risk Classification
Name:AKUL SHARMA  
Roll No:  202401100300027
Date:22/04/2025 

---

## 🎯 Problem Statement
Build a classifier to predict health risk levels (**low/medium/high**) using:
- Body Mass Index (BMI)
- Weekly exercise hours
- Junk food consumption frequency

Dataset: `health_risk.csv` (100 samples)

---

## 🛠 Methodology
### 1. Data Preprocessing
# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Encode labels
y = y.map({'low':0, 'medium':1, 'high':2})
2. Model Implementation (Random Forest)
python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
3. Evaluation Metrics
Metric	Score
Accuracy	75.6%
Precision	75.6%
Recall	75.6%
F1-Score	75.6%
📊 Results
Confusion Matrix
python
[[10  2  0]  # Actual Low
 [ 3 11  2]  # Actual Medium
 [ 1  2 10]] # Actual High
Confusion Matrix Plot

Feature Importance
BMI (65%)

Exercise Hours (20%)

Junk Food Frequency (15%)
