import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
from sklearn.preprocessing import StandardScaler  # Import Scaler for normalization

# Load Data
file_path = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20DataCleaning/AI_PES%20-%20Inferno%20CleanedDataset.xlsx"  # Path to Cleaned Dataset

df = pd.read_excel(file_path)  # Read dataset into DataFrame

print("Initial shape:", df.shape)  # Print dataframe shape

# ---------------- Drop Irrelevant Columns ----------------
drop_cols = ["Patient_ID", "Full_Name", "DOB"]  # Columns to drop
df = df.drop(columns=[c for c in drop_cols if c in df.columns])  # Remove non-predictive columns

# ---------------- Split Blood Pressure ----------------
if 'Blood_Pressure' in df.columns:  # Check if BP exists
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood_Pressure'].str.split('/', expand=True)  # Split BP into Sys/Dia
    df['Systolic_BP'] = df['Systolic_BP'].astype(int)  # Convert Systolic to int
    df['Diastolic_BP'] = df['Diastolic_BP'].astype(int)  # Convert Diastolic to int
    df = df.drop(columns=['Blood_Pressure'])  # Drop original BP column

# ---------------- Ordinal Encoding ----------------
if 'Stages' in df.columns:  # Check if Stages exists
    stage_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}  # Map Stage Roman numerals to integers
    df['Stages'] = df['Stages'].map(stage_map)  # Apply mapping

ordinal_mappings = {  # Define order for ordinal variables
    "Tobacco_Usage": {"None": 0, "Occasional": 1, "Regular": 2},
    "Acholic_Consumption": {"None": 0, "Occasional": 1, "Regular": 2},
    "Physical_Activity": {"Low": 0, "Moderate": 1, "High": 2}
}

for col, mapping in ordinal_mappings.items():  # Iterate mappings
    if col in df.columns:  # Check if column exists
        df[col] = df[col].map(mapping)  # Apply ordinal mapping

# ---------------- Encode Target ----------------
if 'Diagnosis_Status' in df.columns:  # Check for target column
    df['Diagnosis_Status'] = df['Diagnosis_Status'].map({'Negative': 0, 'Positive': 1})  # Binary encode Target

# ---------------- Feature Engineering ----------------
if {'WBC_Count', 'RBC_Count'}.issubset(df.columns):  # Check for blood counts
    df['WBC_RBC_Ratio'] = df['WBC_Count'] / df['RBC_Count']  # Create Ratio feature

if {'Hemoglobin_Level', 'RBC_Count'}.issubset(df.columns):  # Check for Hemo/RBC
    df['Hemoglobin_per_RBC'] = df['Hemoglobin_Level'] / df['RBC_Count']  # Create MCH-like feature

if {'Systolic_BP', 'Diastolic_BP'}.issubset(df.columns):  # Check for BP
    df['BP_Product'] = df['Systolic_BP'] * df['Diastolic_BP']  # Create Interaction term

if {'Tumor Size', 'Age'}.issubset(df.columns):  # Check for Tumor/Age
    df['Tumor_Age_Ratio'] = df['Tumor Size'] / (df['Age'] + 1)  # Create derived ratio

# ---------------- Nominal Categorical Columns ----------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()  # Find remaining categorical cols

if 'Diagnosis_Status' in categorical_cols:  # Exclude target
    categorical_cols.remove('Diagnosis_Status')

# One-hot encode only unordered categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # Apply One-Hot Encoding

# ---------------- Handle Missing Values ----------------
df = df.fillna(df.median(numeric_only=True))  # Fill numeric missing values with median

# ---------------- Z-Score Normalization ----------------
scaler = StandardScaler()  # Initialize Scaler
num_cols = df.select_dtypes(include=np.number).columns.tolist()  # Identify numeric columns
df[num_cols] = scaler.fit_transform(df[num_cols])  # normalize numeric features

# ---------------- Split X and y ----------------
if 'Diagnosis_Status' in df.columns:  # Separating Target
    X = df.drop(columns=['Diagnosis_Status'])
    y = df['Diagnosis_Status']
else:
    X = df
    y = None

# ---------------- Save XLSX ----------------
df.to_excel("AI_PES - Inferno FeatureEngineering/AI_PES - Inferno Feature_Engineered_Data.xlsx", index=False)  # Save engineered data

print("Final shape:", df.shape)  # Print final dimensions
print("Feature Engineering Completed Successfully.")  # Success message
