import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import os  # Import os for file path handling
import re  # Import regex for pattern matching
from datetime import datetime  # Import datetime for date parsing

# --- Configuration ---
# --- Configuration ---
INPUT_FILE = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20AnomalyInjection/AI_PES%20-%20Inferno%20AnomalousDataset.xlsx"  # Input file path (URL)
OUTPUT_FOLDER = r'AI_PES - Inferno DataCleaning'  # Folder for output
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'AI_PES - Inferno CleanedDataset.xlsx')  # Complete output file path


def clean_dataset():
    print("Loading dataset...")  # Notify user
    # Resolve absolute path for robustness
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
    input_path = os.path.join(current_dir, INPUT_FILE)  # Construct local input path (unused if URL)
    df = pd.read_excel(INPUT_FILE)  # Load data into DataFrame
    
    # 1. basic string stripping
    print("Standardizing string columns...")  # Notify user
    for col in df.select_dtypes(include="object").columns:  # Iterate over text columns
        df[col] = df[col].astype(str).str.strip()  # Remove leading/trailing whitespace
        # Convert string 'nan', 'None' to actual NaN
        df[col] = df[col].replace(['nan', 'NaN', 'None', '', 'UNKNOWN', 'unknown'], np.nan)  # Standardize missing values

    # ---------------- DOB & AGE ----------------
    print("Cleaning DOB and Age...")  # Notify user
    
    # Helper to parse DOB
    def parse_dob(x):
        if pd.isna(x): return pd.NaT  # Return missing if input is missing
        # Try multiple formats
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:  # List of date formats to try
            try:
                return pd.to_datetime(x, format=fmt)  # Attempt parsing
            except:
                continue  # Try next format if fail
        # If timestamp, try pandas auto
        try:
             return pd.to_datetime(x)  # Auto-parse fallback
        except:
            return pd.NaT  # Return fail

    df['DOB_Temp'] = df['DOB'].apply(parse_dob)  # Parse DOB column
    
    # Clean Age first (force numeric)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')  # Convert Age to number
    
    # Repair Age/DOB
    today = pd.Timestamp.today()  # Get today's date
    
    for i, row in df.iterrows():  # Iterate rows to fix Age/DOB
        age = row['Age']  # Get Age
        dob = row['DOB_Temp']  # Get parsed DOB
        
        # Valid Age check (0-120)
        is_age_valid = pd.notna(age) and (0 <= age <= 120)  # Check if Age is realistic
        
        # Sync Logic
        if is_age_valid and pd.isna(dob):
            # Estimate DOB from Age
            est_year = today.year - int(age)  # Calculate birth year
            df.at[i, 'DOB_Temp'] = pd.Timestamp(year=est_year, month=1, day=1)  # Set Jan 1st of that year
        elif not is_age_valid and pd.notna(dob):
            # Calculate Age from DOB
            calc_age = (today - dob).days // 365.25  # Calculate age from DOB
            if 0 <= calc_age <= 120:
                df.at[i, 'Age'] = calc_age  # Fill Age if valid
            else:
                # DOB is likely wrong too if it gives weird age
                df.at[i, 'Age'] = np.nan  # Mark as invalid
                df.at[i, 'DOB_Temp'] = pd.NaT  # Mark as invalid
        elif not is_age_valid and pd.isna(dob):
            # Both missing/invalid - force Age to NaN to be imputed
            df.at[i, 'Age'] = np.nan  # Ensure NaN
        elif is_age_valid and pd.notna(dob):
            # Consistency check
            calc_age = (today - dob).days // 365.25  # Recalculate Age
            if abs(calc_age - age) > 2: # Tolerance
                # Trust DOB if reasonable, else trust Age
                if 0 <= calc_age <= 120:
                    df.at[i, 'Age'] = calc_age  # Update Age
                else:
                    est_year = today.year - int(age)  # Re-estimate DOB
                    df.at[i, 'DOB_Temp'] = pd.Timestamp(year=est_year, month=1, day=1)  # Update DOB
    
    # Impute missing Age with Median
    median_age = df['Age'].median()  # Calculate median age
    df['Age'] = df['Age'].fillna(median_age)  # Fill missing ages
    
    # Finalize DOB (Format DD/MM/YYYY)
    # If still NaT, generate from Age
    mask_nat = df['DOB_Temp'].isna()  # Find missing DOBs
    df.loc[mask_nat, 'DOB_Temp'] = df.loc[mask_nat, 'Age'].apply(lambda a: pd.Timestamp(year=int(today.year - a), month=1, day=1))  # Generate from Age
    
    df['DOB'] = df['DOB_Temp'].dt.strftime('%d/%m/%Y')  # Format DOB string
    df.drop(columns=['DOB_Temp'], inplace=True)  # Remove temp column

    # ---------------- GENDER ----------------
    print("Cleaning Gender...")  # Notify user
    def clean_gender(x):
        if pd.isna(x): return np.nan
        s = str(x).lower().strip()
        if s in ['m', 'male']: return 'Male'  # Standardize 'm'
        if s in ['f', 'female']: return 'Female'  # Standardize 'f'
        return np.nan  # Invalid
    
    df['Gender'] = df['Gender'].apply(clean_gender)  # Clean Gender column
    # Impute Gender
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])  # Fill missing with Mode

    # ---------------- BLOOD GROUP ----------------
    print("Cleaning Blood Group...")  # Notify user
    valid_bg = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]  # Valid list
    df['Blood_Group'] = df['Blood_Group'].apply(lambda x: x if x in valid_bg else np.nan)  # Filter invalid
    df['Blood_Group'] = df['Blood_Group'].fillna(df['Blood_Group'].mode()[0])  # Fill missing with Mode

    # ---------------- BLOOD PRESSURE ----------------
    print("Cleaning Blood Pressure...")  # Notify user
    def clean_bp(x):
        if pd.isna(x): return np.nan
        s = str(x)
        # Extract numbers using regex (e.g., 120/80, 120-80, 120 / 80)
        match = re.search(r'(\d{2,3})\D+(\d{2,3})', s)  # Find 2 numbers
        if match:
            sys = int(match.group(1))
            dia = int(match.group(2))
            # Basic physiological checks (Sys > Dia, reasonable range)
            if 50 < sys < 250 and 30 < dia < 150 and sys > dia:
                return f"{sys}/{dia}"  # Return standardized format
        return np.nan  # Invalid

    df['Blood_Pressure'] = df['Blood_Pressure'].apply(clean_bp)  # Clean BP column
    df['Blood_Pressure'] = df['Blood_Pressure'].fillna(df['Blood_Pressure'].mode()[0])  # Fill missing

    # ---------------- LIFESTYLE ----------------
    print("Cleaning Lifestyle Columns...")  # Notify user
    # Map valid values
    lifestyle_maps = {
        'Physical_Activity': (['Low', 'Moderate', 'High'], 'Low'), 
        'Acholic_Consumption': (['None', 'Occasional', 'Regular'], 'None'),
        'Tobacco_Usage': (['None', 'Occasional', 'Regular'], 'None')
    }
    
    for col, (valid, default) in lifestyle_maps.items():  # Iterate lifestyle columns
        if col in df.columns:
            # First clean strings
            df[col] = df[col].astype(str).str.strip().str.capitalize()  # Format text
            
            # 'Nan' from string conversion of np.nan needs to be 'None'
            # Also actual 'None' string
            df[col] = df[col].replace(['Nan', 'Nan', ''], default)  # Fix string NaNs
            
            # Now set anything NOT in valid to NaN (invalid data)
            # unexpected strings that aren't 'None'/'Occasional'/'Regular'
            df.loc[~df[col].isin(valid), col] = np.nan  # Mark invalid
            
            # Finally fill potential real NaNs (from invalid junk) with default (None)
            # Instead of Mode, since 'None' is likely the safe default for these.
            df[col] = df[col].fillna(default)  # Fill missing with default

    # ---------------- NUMERIC LABS ----------------
    print("Cleaning Numeric Lab Values...")  # Notify user
    numeric_cols = ['WBC_Count', 'RBC_Count', 'Platelet_Count', 'Hemoglobin_Level', 'Heart_Rate']  # List of numeric columns
    
    for col in numeric_cols:  # Iterate numeric columns
        if col in df.columns:
            # Force numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to number
            
            median_val = df[col].median()  # Calculate median
            
            
            # Using IQR method for robust outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Replace outliers
            outlier_mask = (df[col] < lower) | (df[col] > upper) | df[col].isna()  # Find outliers
            df.loc[outlier_mask, col] = median_val  # Replace outliers with median

    # ---------------- DIAGNOSIS LOGIC ----------------
    print("Applying Diagnosis Logic...")  # Notify user
    if 'Diagnosis_Status' in df.columns:
        # Normalize Status
        df['Diagnosis_Status'] = df['Diagnosis_Status'].str.capitalize()  # Format text
        # Ensure only Positive/Negative
        df['Diagnosis_Status'] = df['Diagnosis_Status'].apply(lambda x: x if x in ['Positive', 'Negative'] else np.nan)
        df['Diagnosis_Status'] = df['Diagnosis_Status'].fillna(df['Diagnosis_Status'].mode()[0])  # Fill missing

        negative_mask = df['Diagnosis_Status'] == 'Negative'  # Identify negative cases
        positive_mask = df['Diagnosis_Status'] == 'Positive'  # Identify positive cases

        # Negative Rules
        df.loc[negative_mask, 'Cancer_Type'] = "NA"  # Negatives have no cancer type
        df.loc[negative_mask, 'Stages'] = "NA"  # Negatives have no stage
        df.loc[negative_mask, 'Tumor Size'] = "NA"  # Negatives have no tumor

        # Positive Rules
        # Cancer_Type
        valid_cancer = ['Benign', 'Malignant']
        df.loc[positive_mask, 'Cancer_Type'] = df.loc[positive_mask, 'Cancer_Type'].apply(lambda x: x if x in valid_cancer else np.nan)  # Validate type
        df.loc[positive_mask, 'Cancer_Type'] = df.loc[positive_mask, 'Cancer_Type'].fillna(df.loc[positive_mask, 'Cancer_Type'].mode()[0])  # Fill missing
        
        # Stages
        valid_stages = ['I', 'II', 'III', 'IV']
        df.loc[positive_mask, 'Stages'] = df.loc[positive_mask, 'Stages'].apply(lambda x: x if x in valid_stages else np.nan)  # Validate stage
        df.loc[positive_mask, 'Stages'] = df.loc[positive_mask, 'Stages'].fillna(df.loc[positive_mask, 'Stages'].mode()[0])  # Fill missing

        # Tumor Size
        # Clean numeric
        df['Tumor Size'] = pd.to_numeric(df['Tumor Size'], errors='coerce')  # Convert to number
        
        # For positives, must be > 0. If <=0 or NaN, replace with median of positives
        pos_tumor_median = df.loc[positive_mask, 'Tumor Size'].median()  # Get median of positives
        if pd.isna(pos_tumor_median): pos_tumor_median = 1.0 # Fallback
        
        def clean_pos_tumor(x):
            if pd.isna(x) or x <= 0: return pos_tumor_median  # Replace bad values
            return x
        
        df.loc[positive_mask, 'Tumor Size'] = df.loc[positive_mask, 'Tumor Size'].apply(clean_pos_tumor)  # Fix positive tumor sizes
        
        # Convert back to mixed type if needed (since "NA" is string)
        # But 'Tumor Size' column is object now.
        # Ensure 'Tumor Size' for negatives is "NA" explicitly again just in case
        df.loc[negative_mask, 'Tumor Size'] = "NA"  # Ensure negatives are NA

    # ---------------- FINAL CHECK ----------------
    print("Final missing values check...")  # Notify user
    # Fill any remaining NaNs (catch-all)
    for col in df.columns:  # Iterate all columns
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())  # Fill numeric with median
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")  # Fill text with Mode
    
    # Save
    output_path = os.path.join(current_dir, 'AI_PES - Inferno CleanedDataset.xlsx')  # Define output path
    
    df.to_excel(output_path, index=False)  # Save cleaned file
    
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
    clean_dataset()  # Run function