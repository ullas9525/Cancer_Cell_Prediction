import pandas as pd
import numpy as np
import random

# URL to the raw dataset on GitHub
url = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20(PES-AIML-15)_RawDataGeneration/AI_PES%20-%20Inferno%20(PES-AIML-15)_RawDataset.xlsx"

print(f"Loading dataset from: {url}...")

try:
    # Load the dataset directly from the URL
    df = pd.read_excel(url)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Calculate target number of rows to affect (35-45% -> approx 40%)
n_rows = len(df)
n_anomalies = int(n_rows * random.uniform(0.35, 0.45))
print(f"Injecting anomalies into approx {n_anomalies} rows out of {n_rows}...")

# 1. Missing Values (NaN)
# Injecting into Age, Blood_Group, WBC_Count, Blood_Pressure
indices_missing = np.random.choice(df.index, size=n_anomalies // 4, replace=False)
df.loc[indices_missing, 'Age'] = np.nan
df.loc[np.random.choice(df.index, size=n_anomalies // 5, replace=False), 'Blood_Group'] = np.nan
df.loc[np.random.choice(df.index, size=n_anomalies // 5, replace=False), 'WBC_Count'] = np.nan
df.loc[np.random.choice(df.index, size=n_anomalies // 5, replace=False), 'Blood_Pressure'] = None

# 2. Wrong Data Types
# Convert numeric Age to string text
indices_wrong_type = np.random.choice(df.index, size=20, replace=False)
df.loc[indices_wrong_type, 'Age'] = "Thirty-Five"
# Convert WBC_Count to text with units or garbage
indices_wbc_text = np.random.choice(df.index, size=20, replace=False)
df.loc[indices_wbc_text, 'WBC_Count'] = "Normal Range"
# Convert Heart_Rate to string
indices_hr_text = np.random.choice(df.index, size=15, replace=False)
df.loc[indices_hr_text, 'Heart_Rate'] = "High"

# 3. Out-of-Range Values
# Age < 0 or > 120
indices_age_out = np.random.choice(df.index, size=15, replace=False)
df.loc[indices_age_out, 'Age'] = np.random.choice([-5, 150, 200, -1], size=15)
# Extreme WBC values
indices_wbc_out = np.random.choice(df.index, size=15, replace=False)
df.loc[indices_wbc_out, 'WBC_Count'] = np.random.choice([50, 500000, 999999], size=15)
# Platelet count > 900 (assuming units are k/uL or similar logic, making it logically high)
indices_platelet = np.random.choice(df.index, size=15, replace=False)
df.loc[indices_platelet, 'Platelet_Count'] = 1200

# 4. Incorrect Blood Group Labels
bad_blood_labels = ["A++", "0+", "ab+", "C-", "Unknown", "B??"]
indices_blood = np.random.choice(df.index, size=30, replace=False)
df.loc[indices_blood, 'Blood_Group'] = [random.choice(bad_blood_labels) for _ in range(30)]

# 5. Broken Blood Pressure Values
# Formats like "150", "/90", "abc/xyz"
bad_bp = ["150", "/80", "120/", "High", "abc/xyz", "120-80"]
indices_bp = np.random.choice(df.index, size=25, replace=False)
df.loc[indices_bp, 'Blood_Pressure'] = [random.choice(bad_bp) for _ in range(25)]

# 6. Gender Inconsistencies
# Mixing case and adding Unknowns
indices_gender = np.random.choice(df.index, size=40, replace=False)
df.loc[indices_gender, 'Gender'] = [random.choice(["male", "FEMALE", "Unknown", "m", "f"]) for _ in range(40)]

# 7. Duplicate Patient_IDs
# Pick 3 random patients and duplicate their rows at the end of the dataframe
duplicate_rows = df.sample(n=3)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 8. Random Spacing/Formatting Errors
# Add leading/trailing spaces to string columns like Name or Gender (if Name exists, else Gender/Blood Group)
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    indices_space = np.random.choice(df.index, size=10, replace=False)
    # Ensure values are strings before adding space
    df.loc[indices_space, col] = df.loc[indices_space, col].astype(str) + " " 

print("Anomalies injected.")

# Save the final anomalous dataset
output_filename = "/AI_PES - Inferno (PES-AIML-15)_AnomalousDataset.xlsx"
df.to_excel(output_filename, index=False)

print(f"File saved as: {output_filename}")