import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import joblib  # Import joblib for model saving
from sklearn.model_selection import cross_val_score  # Import cross-validation tool
from sklearn.preprocessing import LabelEncoder  # Import Label Encoder
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report, roc_curve, f1_score # Import metrics
from imblearn.over_sampling import SMOTE  # Import SMOTE for balancing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # Import preprocessing tools
from sklearn.ensemble import GradientBoostingClassifier  # Import Gradient Boosting

# 1. Load Data
TRAIN_URL = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20SplittingofData/AI_PES%20-%20Inferno%20Training_data.xlsx"  # Training data URL
TEST_URL = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20SplittingofData/AI_PES%20-%20Inferno%20Testing_data.xlsx"  # Testing data URL

print("Loading Data...")  # Notify user
df_train = pd.read_excel(TRAIN_URL)  # Load training data
df_test = pd.read_excel(TEST_URL)  # Load testing data

leakage_cols = ['Stages', 'Tumor Size', 'Tumor_Age_Ratio', 'Cancer_Type_Malignant']  # List of columns to drop (leakage)

print(f"Dropping Leakage Columns: {leakage_cols}")  # Notify user
df_train = df_train.drop(columns=[c for c in leakage_cols if c in df_train.columns])  # Drop leakage from train
df_test = df_test.drop(columns=[c for c in leakage_cols if c in df_test.columns])  # Drop leakage from test

# Encode Target
le = LabelEncoder()  # Initialize Encoder
df_train['Diagnosis_Status'] = le.fit_transform(df_train['Diagnosis_Status'])  # Encode train target
df_test['Diagnosis_Status'] = le.transform(df_test['Diagnosis_Status'])  # Encode test target

X_train = df_train.drop('Diagnosis_Status', axis=1)  # Features Train
y_train = df_train['Diagnosis_Status']  # Target Train
X_test = df_test.drop('Diagnosis_Status', axis=1)  # Features Test
y_test = df_test['Diagnosis_Status']  # Target Test

# Encode Categorical Features (if any remain)
X_train = pd.get_dummies(X_train, drop_first=True)  # One-Hot Encode Train
X_test = pd.get_dummies(X_test, drop_first=True)  # One-Hot Encode Test

# Align Columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)  # Ensure columns match

# Poly Features (Interaction terms) to find hidden signals
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)  # Initialize Poly Features
X_train_poly = poly.fit_transform(X_train)  # Create interaction terms for train
X_test_poly = poly.transform(X_test)  # Create interaction terms for test

# Scale
scaler = StandardScaler()  # Initialize Scaler
X_train_scaled = scaler.fit_transform(X_train_poly)  # Scale train features
X_test_scaled = scaler.transform(X_test_poly)  # Scale test features

# SMOTE
smote = SMOTE(random_state=42)  # Initialize SMOTE
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)  # Balance training data

# 4. Model Training & Evaluation
models = {  # Define models to train
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=7),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

best_acc = 0  # Track best accuracy
best_model_name = ""  # Track best model name
plt.figure(figsize=(10, 8))  # Initialize plot

for name, model in models.items():  # Iterate through models
    print(f"\nTraining {name}...")  # Notify user
    model.fit(X_train_sm, y_train_sm)  # Train model
    
    # Predictions
    y_pred = model.predict(X_test_scaled)  # Predict classes
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Predict probabilities
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)  # Calculate Accuracy
    roc = roc_auc_score(y_test, y_prob)  # Calculate ROC AUC
    ll = log_loss(y_test, y_prob)  # Calculate Log Loss
    f1 = f1_score(y_test, y_pred)  # Calculate F1 Score
    
    # Cross-Validation (5-fold)
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=5, scoring='accuracy')  # Run Cross-Validation
    cv_acc = cv_scores.mean()  # Average CV score

    print(f"Accuracy: {acc*100:.2f}%")  # Print Accuracy
    print(f"Cross-Validated Accuracy: {cv_acc*100:.2f}%")  # Print CV Accuracy
    print(f"F1 Score: {f1:.4f}")  # Print F1 Score
    print(f"ROC-AUC: {roc:.4f}")  # Print ROC AUC
    print(f"Log Loss: {ll:.4f}")  # Print Log Loss
    print("Confusion Matrix:")  # Print Confusion Matrix header
    print(confusion_matrix(y_test, y_pred))  # Print Confusion Matrix
    print("Classification Report:")  # Print Report header
    print(classification_report(y_test, y_pred))  # Print Report
    
    # Save Model
    joblib.dump(model, f"AI_PES - Inferno ModelTraining/AI_PES - Inferno {name}.pkl")  # Save model file
    
    # Track Best
    if acc > best_acc:  # Check if best
        best_acc = acc  # Update best accuracy
        best_model_name = name  # Update best name
        best_model = model  # Update best model object

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)  # Calculate ROC curve points
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.2f})")  # Plot ROC curve

# Finalize Plot
plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line
plt.xlabel('False Positive Rate')  # X Label
plt.ylabel('True Positive Rate')  # Y Label
plt.title('ROC Curve')  # Title
plt.legend()  # Legend
plt.savefig("AI_PES - Inferno ModelTraining/AI_PES - Inferno ROC_Curve.png")  # Save Plot
print("\nROC Curve saved.")  # Notify user

print(f"\n>>> BEST MODEL: {best_model_name} with Accuracy: {best_acc*100:.2f}% <<<")  # Print Best Model

# Save Predictions to Excel
print("Saving predictions to Excel...")  # Notify user
X_test_final = X_test.copy()  # Copy features
X_test_final['Actual_Diagnosis'] = y_test  # Add actual labels
X_test_final['Predicted_Diagnosis'] = best_model.predict(X_test_scaled)  # Add predicted labels
X_test_final.to_excel("AI_PES - Inferno ModelTraining/AI_PES - Inferno Model_Predictions.xlsx", index=False)  # Save to Excel
print("Predictions saved to 'AI_PES - Inferno ModelTraining/AI_PES - Inferno Model_Predictions.xlsx'")  # Notify user