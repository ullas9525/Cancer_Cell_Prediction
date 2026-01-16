import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report, roc_curve
from imblearn.over_sampling import SMOTE

# 1. Load Data
TRAIN_URL = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20SplittingofData/AI_PES%20-%20Inferno%20Training_data.xlsx"
TEST_URL = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20SplittingofData/AI_PES%20-%20Inferno%20Testing_data.xlsx"

print("Loading Data...")
df_train = pd.read_excel(TRAIN_URL)
df_test = pd.read_excel(TEST_URL)

# 2. Preprocessing
# 2. Preprocessing
# Remove LEAKAGE columns (Post-diagnosis info)
# 'Stages', 'Tumor Size', 'Cancer_Type_Malignant' are direct indicators of the label.
# 'Tumor_Age_Ratio' is derived from Tumor Size.
leakage_cols = ['Stages', 'Tumor Size', 'Tumor_Age_Ratio', 'Cancer_Type_Malignant']

print(f"Dropping Leakage Columns: {leakage_cols}")
df_train = df_train.drop(columns=[c for c in leakage_cols if c in df_train.columns])
df_test = df_test.drop(columns=[c for c in leakage_cols if c in df_test.columns])

# Encode Target
le = LabelEncoder()
df_train['Diagnosis_Status'] = le.fit_transform(df_train['Diagnosis_Status'])
df_test['Diagnosis_Status'] = le.transform(df_test['Diagnosis_Status'])

X_train = df_train.drop('Diagnosis_Status', axis=1)
y_train = df_train['Diagnosis_Status']
X_test = df_test.drop('Diagnosis_Status', axis=1)
y_test = df_test['Diagnosis_Status']

# Encode Categorical Features (if any remain)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align Columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 3. Feature Engineering & SMOTE
print("Applying Interaction Features & SMOTE...")
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Poly Features (Interaction terms) to find hidden signals
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

# 4. Model Training & Evaluation
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=7),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

best_acc = 0
best_model_name = ""
plt.figure(figsize=(10, 8))

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_sm, y_train_sm)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    joblib.dump(model, f"AI_PES - Inferno ModelTraining/AI_PES - Inferno {name}.pkl")
    
    # Track Best
    if acc > best_acc:
        best_acc = acc
        best_model_name = name

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.2f})")

# Finalize Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("AI_PES - Inferno ModelTraining/AI_PES - Inferno ROC_Curve.png")
print("\nROC Curve saved.")

print(f"\n>>> BEST MODEL: {best_model_name} with Accuracy: {best_acc*100:.2f}% <<<")
