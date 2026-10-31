"""
Breast Cancer Detection using Machine Learning
Educational classification project based on breast_cancer.csv.
Target: diagnosis (B = Benign, M = Malignant)
Important: This model is for academic demonstration only, not medical diagnosis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

#DATA_PATH = os.path.join("Dataset","breast_cancer.csv")
REPORT_DIR = "Reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv("breast_cancer.csv")
print("\\n--- Dataset Overview ---")
print("Shape:", df.shape)
print(df.head())
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\\nDiagnosis counts:\\n", df["diagnosis"].value_counts())

# 2. Define target and predictors; ID is an identifier, not a predictive feature
X = df.drop(columns=["diagnosis", "id"])
y = df["diagnosis"].map({"B": 0, "M": 1})
if y.isnull().any():
    raise ValueError("Unexpected diagnosis values. Expected B and M.")

# 3. Exploratory Data Analysis (EDA)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="diagnosis")
plt.title("Diagnosis Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "diagnosis_distribution.png"), dpi=150)
plt.close()

for column in ["radius_mean", "texture_mean", "area_mean"]:
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x=column, hue="diagnosis", kde=True, element="step")
    plt.title(f"Distribution of {column} by Diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"{column}_histogram.png"), dpi=150)
    plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=["id"]).select_dtypes(include=np.number).corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "correlation_matrix.png"), dpi=150)
plt.close()

# 4. Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Pipeline: median imputation, scaling, Logistic Regression
# Pipeline ensures preprocessing is learned only from training data.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, C=1.0))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Evaluation
print("\\n--- Test Performance ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print("\\nClassification report:\\n",
      classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# 7. Improvement / validation: 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
print("5-fold CV accuracy scores:", np.round(cv_scores, 4))
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

# 8. Save prediction results for review
results = X_test.copy()
results["Actual_Diagnosis"] = y_test.map({0: "B", 1: "M"})
results["Predicted_Diagnosis"] = pd.Series(y_pred, index=X_test.index).map({0: "B", 1: "M"})
results.to_csv(os.path.join(REPORT_DIR, "test_predictions.csv"), index=False)
print("\\nFiles saved in Reports/.")
print("Reminder: Academic demonstration only; not for clinical use.")
