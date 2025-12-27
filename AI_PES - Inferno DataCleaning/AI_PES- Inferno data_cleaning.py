import pandas as pd
import numpy as np

# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------
DATA_URL = ("https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20AnomalyInjection/AI_PES%20-%20Inferno%20AnomalousDataset.xlsx"
)

df = pd.read_excel(DATA_URL)
print("Original shape:", df.shape)

# -------------------------------------------------
# 2. Strip column name spaces (DO NOT rename)
# -------------------------------------------------
df.columns = df.columns.str.strip()

# -------------------------------------------------
# 3. Safe column finder (NO IndexError possible)
# -------------------------------------------------
def find_col(contains_any):
    for col in df.columns:
        name = col.lower()
        if any(key in name for key in contains_any):
            return col
    return None

age_col        = find_col(["age"])
gender_col     = find_col(["gender"])
blood_group_col= find_col(["blood group", "bloodgroup"])
wbc_col        = find_col(["wbc"])
rbc_col        = find_col(["rbc"])
platelet_col   = find_col(["platelet"])
hb_col         = find_col(["hemoglobin", "hb"])
bp_col         = find_col(["blood pressure", "bp"])

# -------------------------------------------------
# 4. Remove duplicates
# -------------------------------------------------
df = df.drop_duplicates()

# -------------------------------------------------
# 5. Clean AGE (numbers only)
# -------------------------------------------------
def clean_age(val):
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "old":
            return np.nan
        if v == "forty":
            return 40
    try:
        return int(val)
    except:
        return np.nan

df[age_col] = df[age_col].apply(clean_age)
df[age_col] = df[age_col].fillna(df[age_col].median())
df = df[(df[age_col] > 0) & (df[age_col] < 120)]

# -------------------------------------------------
# 6. Clean GENDER (keep Male/Female)
# -------------------------------------------------
df = df[df[gender_col].isin(["Male", "Female"])]

# -------------------------------------------------
# 7. Clean BLOOD GROUP (keep as is)
# -------------------------------------------------
valid_blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
df = df[df[blood_group_col].isin(valid_blood_groups)]

# -------------------------------------------------
# 8. Clean WBC Count
# -------------------------------------------------
def clean_wbc(val):
    if isinstance(val, str):
        v = val.lower()
        if v == "suppressed":
            return 3.5
        if v == "normal":
            return 6.0
    return val

df[wbc_col] = df[wbc_col].apply(clean_wbc)
df[wbc_col] = pd.to_numeric(df[wbc_col], errors="coerce")
df[wbc_col] = df[wbc_col].fillna(df[wbc_col].median())

# -------------------------------------------------
# 9. Clean numeric columns
# -------------------------------------------------
for col in [rbc_col, platelet_col, hb_col]:
    if col:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# -------------------------------------------------
# 10. Split BLOOD PRESSURE (ONLY if present)
# -------------------------------------------------
if bp_col:
    def split_bp(val):
        if isinstance(val, str) and "/" in val:
            try:
                s, d = val.split("/")
                return pd.Series([int(s), int(d)])
            except:
                return pd.Series([np.nan, np.nan])
        return pd.Series([np.nan, np.nan])

    df[["Systolic_BP", "Diastolic_BP"]] = df[bp_col].apply(split_bp)
    df.drop(columns=[bp_col], inplace=True)
else:
    print("⚠️ Blood Pressure column not found — skipping BP split")

# -------------------------------------------------
# 11. Final cleanup
# -------------------------------------------------
df = df.dropna()
print("Cleaned shape:", df.shape)

# -------------------------------------------------
# 12. Save cleaned dataset
# -------------------------------------------------
df.to_excel(
    "AI_PES - Inferno DataCleaning/Cleaned_Cancer_Dataset.xlsx",
    index=False,
    engine="openpyxl"
)

print("✅ Dataset cleaned correctly")
print("✅ Age numeric, Gender & Blood Group preserved")
