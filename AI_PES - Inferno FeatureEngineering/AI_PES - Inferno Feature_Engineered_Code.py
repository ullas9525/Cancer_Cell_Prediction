import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

file_path = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20DataCleaning/AI_PES%20-%20Inferno%20CleanedDataset.xlsx"

df = pd.read_excel(file_path)

print("Initial shape:", df.shape)

# ---------------- Drop Irrelevant Columns ----------------
drop_cols = ["Patient_ID", "Full_Name", "DOB"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---------------- Split Blood Pressure ----------------
if 'Blood_Pressure' in df.columns:
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = df['Systolic_BP'].astype(int)
    df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)
    df = df.drop(columns=['Blood_Pressure'])

# ---------------- Ordinal Encoding ----------------
if 'Stages' in df.columns:
    stage_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    df['Stages'] = df['Stages'].map(stage_map)

ordinal_mappings = {
    "Tobacco_Usage": {"None": 0, "Occasional": 1, "Regular": 2},
    "Acholic_Consumption": {"None": 0, "Occasional": 1, "Regular": 2},
    "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2}
}

for col, mapping in ordinal_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# ---------------- Encode Target ----------------
if 'Diagnosis_Status' in df.columns:
    df['Diagnosis_Status'] = df['Diagnosis_Status'].map({'Negative': 0, 'Positive': 1})

# ---------------- Feature Engineering ----------------
if {'WBC_Count', 'RBC_Count'}.issubset(df.columns):
    df['WBC_RBC_Ratio'] = df['WBC_Count'] / df['RBC_Count']

if {'Hemoglobin_Level', 'RBC_Count'}.issubset(df.columns):
    df['Hemoglobin_per_RBC'] = df['Hemoglobin_Level'] / df['RBC_Count']

if {'Systolic_BP', 'Diastolic_BP'}.issubset(df.columns):
    df['BP_Product'] = df['Systolic_BP'] * df['Diastolic_BP']

if {'Tumor Size', 'Age'}.issubset(df.columns):
    df['Tumor_Age_Ratio'] = df['Tumor Size'] / (df['Age'] + 1)

# ---------------- Nominal Categorical Columns ----------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if 'Diagnosis_Status' in categorical_cols:
    categorical_cols.remove('Diagnosis_Status')

# One-hot encode only unordered categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ---------------- Handle Missing Values ----------------
df = df.fillna(df.median(numeric_only=True))

# ---------------- Z-Score Normalization ----------------
scaler = StandardScaler()
num_cols = df.select_dtypes(include=np.number).columns.tolist()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ---------------- Split X and y ----------------
if 'Diagnosis_Status' in df.columns:
    X = df.drop(columns=['Diagnosis_Status'])
    y = df['Diagnosis_Status']
else:
    X = df
    y = None

# ---------------- Save XLSX ----------------
df.to_excel("AI_PES - Inferno FeatureEngineering/AI_PES - Inferno Feature_Engineered_Data.xlsx", index=False)

print("Final shape:", df.shape)
print("Feature Engineering Completed Successfully.")
