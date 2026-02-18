import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting
import joblib  # Model saving
from sklearn.model_selection import GridSearchCV, cross_val_score  # Hyperparameter tuning
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Preprocessing
from sklearn.decomposition import PCA  # Dimensionality reduction
from sklearn.linear_model import LogisticRegression  # Model: LR
from sklearn.tree import DecisionTreeClassifier  # Model: DT
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Model: RF, GB
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report, roc_curve, f1_score  # Metrics
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference, Series
from openpyxl.utils import get_column_letter

# ---------------- CONFIGURATION ----------------
TRAIN_URL = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20SplittingofData/AI_PES%20-%20Inferno%20Training_data.xlsx"
TEST_URL = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20SplittingofData/AI_PES%20-%20Inferno%20Testing_data.xlsx"
OUTPUT_DIR = "AI_PES - Inferno OptimizedCode"

# ----------------- 1. LOAD DATA -----------------
print("Loading data from GitHub...")
df_train = pd.read_excel(TRAIN_URL)  # Load train data
df_test = pd.read_excel(TEST_URL)  # Load test data

# ----------------- 2. DATA PREPROCESSING -----------------
leakage_cols = ['Stages', 'Tumor Size', 'Tumor_Age_Ratio', 'Cancer_Type_Malignant']  # Columns to drop
print(f"Dropping leakage columns: {leakage_cols}")

df_train = df_train.drop(columns=[c for c in leakage_cols if c in df_train.columns])  # Drop from train
df_test = df_test.drop(columns=[c for c in leakage_cols if c in df_test.columns])  # Drop from test

# Encode Target
le = LabelEncoder()
df_train['Diagnosis_Status'] = le.fit_transform(df_train['Diagnosis_Status'])  # Encode target
df_test['Diagnosis_Status'] = le.transform(df_test['Diagnosis_Status'])  # Encode target (consistent)

# Separate Features and Target
X_train = df_train.drop('Diagnosis_Status', axis=1)
y_train = df_train['Diagnosis_Status']
X_test = df_test.drop('Diagnosis_Status', axis=1)
y_test = df_test['Diagnosis_Status']

# One-Hot Encoding for Categorical Features
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align Columns (Ensure Train/Test match)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit & Transform Train
X_test_scaled = scaler.transform(X_test)  # Transform Test

# ----------------- 3. DIMENSIONALITY REDUCTION (PCA) -----------------
print("Applying PCA...")
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA Reduced dimensions from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}")

# ----------------- 4. MODEL TRAINING & OPTIMIZATION -----------------
models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=2000),
        "params": {"C": [0.1, 1, 10]}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {"max_depth": [3, 5, 7, 10], "min_samples_split": [2, 5]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]}
    }
}

best_overall_score = 0
best_overall_acc = 0
best_overall_model_name = ""
best_overall_model = None
results = []

plt.figure(figsize=(10, 8))  # Setup ROC Plot

for name, config in models.items():
    print(f"\nTraining {name} with GridSearchCV...")
    
    # Grid Search
    grid = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_pca, y_train)
    
    best_model = grid.best_estimator_  # Get best model from grid
    
    # Predictions
    y_pred = best_model.predict(X_test_pca)
    y_prob = best_model.predict_proba(X_test_pca)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cv_acc = grid.best_score_  # Best CV score from training
    
    print(f"  Best Params: {grid.best_params_}")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  CV Accuracy: {cv_acc*100:.2f}%")
    print(f"  ROC-AUC: {roc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("  Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Append Results
    results.append({
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": roc,
        "F1 Score": f1,
        "Log Loss": ll
    })
    
    # Store results
    if roc > best_overall_score:  # Selection criteria: ROC-AUC
        best_overall_score = roc
        best_overall_acc = acc
        best_overall_model_name = name
        best_overall_model = best_model

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.2f})")

# ----------------- 5. FINALIZATION -----------------
# Save ROC Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
roc_path = f"{OUTPUT_DIR}/AI_PES - Inferno ROC_Curve.png"
plt.savefig(roc_path)
print(f"\nROC Curve saved to: {roc_path}")

# Save Best Model
model_path = f"{OUTPUT_DIR}/AI_PES - Inferno Best_Model.pkl"
joblib.dump(best_overall_model, model_path)
print(f"Best Model ({best_overall_model_name}) saved to: {model_path}")

# Save Predictions
# ----------------- 6. VISUALIZATION (CHARTS) -----------------
print("Generating Visualization Charts in Excel...")

# Output Path
pred_path = f"{OUTPUT_DIR}/AI_PES - Inferno FinalPredictionChart.xlsx"

# Create Metrics DataFrame
metrics_df = pd.DataFrame(results)

# Create Prediction DataFrame (Re-added)
X_test_final = X_test.copy()
X_test_final['Actual_Diagnosis'] = y_test
X_test_final['Predicted_Diagnosis'] = best_overall_model.predict(X_test_pca)
X_test_final['Prediction_Probability'] = best_overall_model.predict_proba(X_test_pca)[:, 1]

# Create Single Sheet layout
with pd.ExcelWriter(pred_path, engine='openpyxl') as writer:
    # Write Predictions data (Left Side)
    X_test_final.to_excel(writer, sheet_name='Final Analysis', index=False, startrow=0)
    
    # Calculate offset for Metrics Table
    offset = len(X_test_final.columns) + 2
    
    # Write Metrics Table (Right Side)
    metrics_df.to_excel(writer, sheet_name='Final Analysis', index=False, startrow=0, startcol=offset)

# Open Workbook to add Charts
wb = load_workbook(pred_path)
ws = wb['Final Analysis']

# Create Line Chart
chart = LineChart()
chart.title = "Actual vs Predicted"
chart.style = 13  # Standard line style
chart.y_axis.title = 'Diagnosis (0=Neg, 1=Pos)'
chart.x_axis.title = 'Patient Index'

# Define data for Actual and Predicted Columns
total_cols = len(X_test_final.columns)

# Create Reference for Actual Data (Blue)
actual_data = Reference(ws, min_col=total_cols-2, min_row=1, max_col=total_cols-2, max_row=len(X_test_final)+1)
series_actual = Series(actual_data, title="Actual")
series_actual.graphicalProperties.line.solidFill = "0000FF"  # Blue Color

# Create Reference for Predicted Data (Red)
predicted_data = Reference(ws, min_col=total_cols-1, min_row=1, max_col=total_cols-1, max_row=len(X_test_final)+1)
series_predicted = Series(predicted_data, title="Predicted")
series_predicted.graphicalProperties.line.solidFill = "FF0000"  # Red Color

# Add Series to Chart
chart.series.append(series_actual)
chart.series.append(series_predicted)

# Place Chart below Metrics Table (Same column offset)
metrics_start_col = offset + 1
chart_cell = get_column_letter(metrics_start_col) + "10"
ws.add_chart(chart, chart_cell)

wb.save(pred_path)
# Final Summary
print(f"\n>>> BEST MODEL: {best_overall_model_name} with Accuracy: {best_overall_acc*100:.2f}% <<<")