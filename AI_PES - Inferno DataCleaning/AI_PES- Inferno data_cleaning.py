import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# --- Configuration ---
# --- Configuration ---
INPUT_FILE = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20AnomalyInjection/AI_PES%20-%20Inferno%20AnomalousDataset.xlsx"
OUTPUT_FOLDER = r'AI_PES - Inferno DataCleaning'
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'AI_PES - Inferno CleanedDataset.xlsx')


def clean_dataset():
    print("Loading dataset...")
    # Resolve absolute path for robustness
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, INPUT_FILE)
    df = pd.read_excel(INPUT_FILE)
    
    # 1. basic string stripping
    print("Standardizing string columns...")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        # Convert string 'nan', 'None' to actual NaN
        df[col] = df[col].replace(['nan', 'NaN', 'None', '', 'UNKNOWN', 'unknown'], np.nan)

    # ---------------- DOB & AGE ----------------
    print("Cleaning DOB and Age...")
    
    # Helper to parse DOB
    def parse_dob(x):
        if pd.isna(x): return pd.NaT
        # Try multiple formats
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                return pd.to_datetime(x, format=fmt)
            except:
                continue
        # If timestamp, try pandas auto
        try:
             return pd.to_datetime(x)
        except:
            return pd.NaT

    df['DOB_Temp'] = df['DOB'].apply(parse_dob)
    
    # Clean Age first (force numeric)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    # Repair Age/DOB
    today = pd.Timestamp.today()
    
    for i, row in df.iterrows():
        age = row['Age']
        dob = row['DOB_Temp']
        
        # Valid Age check (0-120)
        is_age_valid = pd.notna(age) and (0 <= age <= 120)
        
        # Sync Logic
        if is_age_valid and pd.isna(dob):
            # Estimate DOB from Age
            est_year = today.year - int(age)
            df.at[i, 'DOB_Temp'] = pd.Timestamp(year=est_year, month=1, day=1)
        elif not is_age_valid and pd.notna(dob):
            # Calculate Age from DOB
            calc_age = (today - dob).days // 365.25
            if 0 <= calc_age <= 120:
                df.at[i, 'Age'] = calc_age
            else:
                # DOB is likely wrong too if it gives weird age
                df.at[i, 'Age'] = np.nan
                df.at[i, 'DOB_Temp'] = pd.NaT
        elif not is_age_valid and pd.isna(dob):
            # Both missing/invalid - force Age to NaN to be imputed
            df.at[i, 'Age'] = np.nan
        elif is_age_valid and pd.notna(dob):
            # Consistency check
            calc_age = (today - dob).days // 365.25
            if abs(calc_age - age) > 2: # Tolerance
                # Trust DOB if reasonable, else trust Age
                if 0 <= calc_age <= 120:
                    df.at[i, 'Age'] = calc_age
                else:
                    est_year = today.year - int(age)
                    df.at[i, 'DOB_Temp'] = pd.Timestamp(year=est_year, month=1, day=1)
    
    # Impute missing Age with Median
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    
    # Finalize DOB (Format DD/MM/YYYY)
    # If still NaT, generate from Age
    mask_nat = df['DOB_Temp'].isna()
    df.loc[mask_nat, 'DOB_Temp'] = df.loc[mask_nat, 'Age'].apply(lambda a: pd.Timestamp(year=int(today.year - a), month=1, day=1))
    
    df['DOB'] = df['DOB_Temp'].dt.strftime('%d/%m/%Y')
    df.drop(columns=['DOB_Temp'], inplace=True)

    # ---------------- GENDER ----------------
    print("Cleaning Gender...")
    def clean_gender(x):
        if pd.isna(x): return np.nan
        s = str(x).lower().strip()
        if s in ['m', 'male']: return 'Male'
        if s in ['f', 'female']: return 'Female'
        return np.nan
    
    df['Gender'] = df['Gender'].apply(clean_gender)
    # Impute Gender
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

    # ---------------- BLOOD GROUP ----------------
    print("Cleaning Blood Group...")
    valid_bg = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    df['Blood_Group'] = df['Blood_Group'].apply(lambda x: x if x in valid_bg else np.nan)
    df['Blood_Group'] = df['Blood_Group'].fillna(df['Blood_Group'].mode()[0])

    # ---------------- BLOOD PRESSURE ----------------
    print("Cleaning Blood Pressure...")
    def clean_bp(x):
        if pd.isna(x): return np.nan
        s = str(x)
        # Extract numbers using regex (e.g., 120/80, 120-80, 120 / 80)
        match = re.search(r'(\d{2,3})\D+(\d{2,3})', s)
        if match:
            sys = int(match.group(1))
            dia = int(match.group(2))
            # Basic physiological checks (Sys > Dia, reasonable range)
            if 50 < sys < 250 and 30 < dia < 150 and sys > dia:
                return f"{sys}/{dia}"
        return np.nan

    df['Blood_Pressure'] = df['Blood_Pressure'].apply(clean_bp)
    df['Blood_Pressure'] = df['Blood_Pressure'].fillna(df['Blood_Pressure'].mode()[0])

    # ---------------- LIFESTYLE ----------------
    print("Cleaning Lifestyle Columns...")
    # Map valid values
    lifestyle_maps = {
        'Physical_Activity': (['Low', 'Moderate', 'High'], 'Low'), # Default/Mode
        'Acholic_Consumption': (['None', 'Occasional', 'Regular'], 'None'),
        'Tobacco_Usage': (['None', 'Occasional', 'Regular'], 'None')
    }
    
    for col, (valid, default) in lifestyle_maps.items():
        if col in df.columns:
            # Normalize first
            df[col] = df[col].astype(str).str.capitalize()
            # Set invalid to NaN then fill
            df.loc[~df[col].isin(valid), col] = np.nan
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else default)

    # ---------------- NUMERIC LABS ----------------
    print("Cleaning Numeric Lab Values...")
    numeric_cols = ['WBC_Count', 'RBC_Count', 'Platelet_Count', 'Hemoglobin_Level', 'Heart_Rate']
    
    for col in numeric_cols:
        if col in df.columns:
            # Force numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            median_val = df[col].median()
            
            
            # Using IQR method for robust outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Replace outliers
            outlier_mask = (df[col] < lower) | (df[col] > upper) | df[col].isna()
            df.loc[outlier_mask, col] = median_val

    # ---------------- DIAGNOSIS LOGIC ----------------
    print("Applying Diagnosis Logic...")
    if 'Diagnosis_Status' in df.columns:
        # Normalize Status
        df['Diagnosis_Status'] = df['Diagnosis_Status'].str.capitalize()
        # Ensure only Positive/Negative
        df['Diagnosis_Status'] = df['Diagnosis_Status'].apply(lambda x: x if x in ['Positive', 'Negative'] else np.nan)
        df['Diagnosis_Status'] = df['Diagnosis_Status'].fillna(df['Diagnosis_Status'].mode()[0])

        negative_mask = df['Diagnosis_Status'] == 'Negative'
        positive_mask = df['Diagnosis_Status'] == 'Positive'

        # Negative Rules
        df.loc[negative_mask, 'Cancer_Type'] = "NA"
        df.loc[negative_mask, 'Stages'] = "NA"
        df.loc[negative_mask, 'Tumor Size'] = "NA"

        # Positive Rules
        # Cancer_Type
        valid_cancer = ['Benign', 'Malignant']
        df.loc[positive_mask, 'Cancer_Type'] = df.loc[positive_mask, 'Cancer_Type'].apply(lambda x: x if x in valid_cancer else np.nan)
        df.loc[positive_mask, 'Cancer_Type'] = df.loc[positive_mask, 'Cancer_Type'].fillna(df.loc[positive_mask, 'Cancer_Type'].mode()[0])
        
        # Stages
        valid_stages = ['I', 'II', 'III', 'IV']
        df.loc[positive_mask, 'Stages'] = df.loc[positive_mask, 'Stages'].apply(lambda x: x if x in valid_stages else np.nan)
        df.loc[positive_mask, 'Stages'] = df.loc[positive_mask, 'Stages'].fillna(df.loc[positive_mask, 'Stages'].mode()[0])

        # Tumor Size
        # Clean numeric
        df['Tumor Size'] = pd.to_numeric(df['Tumor Size'], errors='coerce')
        
        # For positives, must be > 0. If <=0 or NaN, replace with median of positives
        pos_tumor_median = df.loc[positive_mask, 'Tumor Size'].median()
        if pd.isna(pos_tumor_median): pos_tumor_median = 1.0 # Fallback
        
        def clean_pos_tumor(x):
            if pd.isna(x) or x <= 0: return pos_tumor_median
            return x
        
        df.loc[positive_mask, 'Tumor Size'] = df.loc[positive_mask, 'Tumor Size'].apply(clean_pos_tumor)
        
        # Convert back to mixed type if needed (since "NA" is string)
        # But 'Tumor Size' column is object now.
        # Ensure 'Tumor Size' for negatives is "NA" explicitly again just in case
        df.loc[negative_mask, 'Tumor Size'] = "NA"

    # ---------------- FINAL CHECK ----------------
    print("Final missing values check...")
    # Fill any remaining NaNs (catch-all)
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    # Save
    output_path = os.path.join(current_dir, 'AI_PES - Inferno CleanedDataset.xlsx')
    
    df.to_excel(output_path, index=False)
    
    print("------------------------------------------------")
    print("CLEAN DATASET SAVED SUCCESSFULLY")
    print(f"Location: {output_path}")
    print("------------------------------------------------")
    # Quick Validation Print
    print("\nValidation Summary:")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"Unique Gender: {df['Gender'].unique()}")
    print(f"Unique Blood_Group: {df['Blood_Group'].unique()}")
    if "Diagnosis_Status" in df.columns:
         print(f"Unique Diagnosis_Status: {df['Diagnosis_Status'].unique()}")

if __name__ == "__main__":
    clean_dataset()