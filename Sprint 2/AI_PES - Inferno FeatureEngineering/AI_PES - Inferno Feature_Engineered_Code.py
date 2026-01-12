import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

file_path = "/content/AI_PES - Inferno CleanedDataset (2).xlsx"
df = pd.read_excel(file_path)

print("Initial shape:", df.shape)

drop_cols = ["Patient_ID", "Full_Name", "DOB"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

if 'Blood_Pressure' in df.columns:
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = df['Systolic_BP'].astype(int)
    df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)
    df = df.drop(columns=['Blood_Pressure'])

if 'Stages' in df.columns:
    stage_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    df['Stages'] = df['Stages'].map(stage_map)

if 'Diagnosis_Status' in df.columns:
    df['Diagnosis_Status'] = df['Diagnosis_Status'].map({'Negative': 0, 'Positive': 1})

if {'WBC_Count', 'RBC_Count'}.issubset(df.columns):
    df['WBC_RBC_Ratio'] = df['WBC_Count'] / df['RBC_Count']

if {'Hemoglobin_Level', 'RBC_Count'}.issubset(df.columns):
    df['Hemoglobin_per_RBC'] = df['Hemoglobin_Level'] / df['RBC_Count']

if {'Systolic_BP', 'Diastolic_BP'}.issubset(df.columns):
    df['BP_Product'] = df['Systolic_BP'] * df['Diastolic_BP']

if {'Tumor Size', 'Age'}.issubset(df.columns):
    df['Tumor_Age_Ratio'] = df['Tumor Size'] / (df['Age'] + 1)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove target if mistakenly included
if 'Diagnosis_Status' in categorical_cols:
    categorical_cols.remove('Diagnosis_Status')

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df = df.fillna(df.median(numeric_only=True))
scaler = StandardScaler()
num_cols = df.select_dtypes(include=np.number).columns.tolist()

df[num_cols] = scaler.fit_transform(df[num_cols])

if 'Diagnosis_Status' in df.columns:
    X = df.drop(columns=['Diagnosis_Status'])
    y = df['Diagnosis_Status']
else:
    X = df
    y = None

df.to_csv("feature_engineered_data.csv", index=False)

print("Final shape:", df.shape)
print("Feature Engineering Completed Successfully.")

