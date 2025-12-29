import pandas as pd
import numpy as np
import os
import re

# Define the input URL and output path
input_url = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20AnomalyInjection/AI_PES%20-%20Inferno%20AnomalousDataset.xlsx"
output_folder = "AI_PES - Inferno DataCleaning"
output_file = f"{output_folder}/AI_PES - Inferno CleanedDataset.xlsx"

print(f"Loading anomalous dataset from: {input_url}...")

# Load the dataset
try:
    df = pd.read_excel(input_url)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Remove duplicate Patient_IDs (keep first occurrence)
df.drop_duplicates(subset='Patient_ID', keep='first', inplace=True)

# SAFE STRIPPING: Only apply strip to actual string values column by column
# This prevents deleting numbers in mixed-type columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

# 1. Fix Gender values
# STRATEGY: Convert to lowercase first to catch "Male", "male", "MALE", "m" all at once
# This prevents dropping valid "Male"/"Female" values that weren't in the specific map keys
df['Gender'] = df['Gender'].astype(str).str.lower().str.strip()
gender_map = {
    'male': 'Male',
    'm': 'Male',
    'female': 'Female', 
    'f': 'Female',
    # Map unknown/nan to actual np.nan so mode imputation handles it later
    'unknown': np.nan, 
    'nan': np.nan
}
# Map values. Unmapped become NaN. Explicitly set unknown/nan keys to NaN too.
# We REMOVED .fillna('Unknown') so that Mode imputation can fill these later.
df['Gender'] = df['Gender'].map(gender_map)

# 2. Fix Numeric Columns & Wrong Data Types
# PROBLEM SOLVER: This function removes text (like 'yrs', 'bpm', ',') before converting
def clean_numeric_noise(val):
    if pd.isna(val): return np.nan
    # Convert to string, then keep digits, dots, and negative signs
    val_str = str(val)
    clean_val = re.sub(r'[^\d.-]', '', val_str)
    
    # Check if multiple dots exist (e.g. 12.3.4), keep first part
    if clean_val.count('.') > 1:
        clean_val = clean_val.split('.')[0] + '.' + clean_val.split('.')[1]
        
    return clean_val if clean_val else np.nan

numeric_cols = ['Age', 'WBC_Count', 'Heart_Rate', 'Platelet_Count']

# Check if columns exist to prevent KeyError
existing_num_cols = [c for c in numeric_cols if c in df.columns]

for col in existing_num_cols:
    # First, scrub the text noise
    df[col] = df[col].apply(clean_numeric_noise)
    # Then convert to number
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Handle Age Logic
# Replace impossible ages (<0 or >120) with NaN
if 'Age' in df.columns:
    df.loc[(df['Age'] < 0) | (df['Age'] > 120), 'Age'] = np.nan

# 4. Standardize Blood Group
# Regex to keep only valid patterns (A, B, AB, O followed by + or -)
valid_bg = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
df.loc[~df['Blood_Group'].isin(valid_bg), 'Blood_Group'] = np.nan

# 5. Fix Blood Pressure
# Function to validate systolic/diastolic format (e.g., 120/80)
def clean_bp(val):
    if pd.isna(val): return np.nan
    match = re.match(r'^(\d{2,3})/(\d{2,3})$', str(val))
    return val if match else np.nan

if 'Blood_Pressure' in df.columns:
    df['Blood_Pressure'] = df['Blood_Pressure'].apply(clean_bp)

# 6. Remove Extreme Outliers
# Using a logical cap for medical data to remove extreme anomalies
if 'WBC_Count' in df.columns:
    df.loc[df['WBC_Count'] > 50000, 'WBC_Count'] = np.nan
if 'Platelet_Count' in df.columns:
    df.loc[df['Platelet_Count'] > 900, 'Platelet_Count'] = np.nan
if 'Heart_Rate' in df.columns:
    df.loc[df['Heart_Rate'] > 220, 'Heart_Rate'] = np.nan

# 7. Impute Missing Values
# Fill numeric columns with Mean
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    mean_val = df[col].mean()
    # Fallback to 0 if mean is NaN (e.g. empty column), preventing fillna failure
    if pd.isna(mean_val):
        mean_val = 0
    df[col] = df[col].fillna(mean_val)

# Global "Unknown" cleanup: Ensure literal "Unknown" strings in ANY column are treated as NaN
df.replace(['Unknown', 'unknown'], np.nan, inplace=True)

# Fill categorical columns with Mode (most frequent value)
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# Ensure correct data types (int for counts/age)
# Round first to handle float means (e.g. 35.6 -> 36) before casting to int
if 'Age' in df.columns:
    df['Age'] = df['Age'].round().astype(int)
if 'Heart_Rate' in df.columns:
    df['Heart_Rate'] = df['Heart_Rate'].round().astype(int)
if 'WBC_Count' in df.columns:
    # Round WBC to 1 decimal place as requested
    df['WBC_Count'] = df['WBC_Count'].astype(float).round(1)

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the cleaned dataset
try:
    df.to_excel(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
except PermissionError:
    print(f"\nERROR: Could not save the file because '{output_file}' is open.")
    print("ACTION: Please CLOSE the Excel file and run the script again.")
except Exception as e:
    print(f"An error occurred while saving: {e}")