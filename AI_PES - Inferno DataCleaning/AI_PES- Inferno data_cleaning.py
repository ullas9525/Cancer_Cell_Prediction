import pandas as pd
import numpy as np
import os
import re

input_url = "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20AnomalyInjection/AI_PES%20-%20Inferno%20AnomalousDataset.xlsx"
output_folder = "AI_PES - Inferno DataCleaning"
output_file = f"{output_folder}/AI_PES - Inferno CleanedDataset.xlsx"

print("Loading dataset...")
df = pd.read_excel(input_url)

df.drop_duplicates(subset="Patient_ID", inplace=True)

# ---------------- BASIC STRIP ----------------
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x,str) else x)

# ---------------- GENDER ----------------
df["Gender"] = df["Gender"].astype(str).str.lower().str.strip()
df["Gender"] = df["Gender"].map({"male":"Male","m":"Male","female":"Female","f":"Female"})

# ---------------- TOBACCO ----------------
if "Tobacco_Usage" in df.columns:
    df["Tobacco_Usage"] = df["Tobacco_Usage"].astype(str).str.strip().str.capitalize()
    df.loc[~df["Tobacco_Usage"].isin(["Regular","Occasional","None"]), "Tobacco_Usage"] = np.nan

# ---------------- NUMERIC CLEAN ----------------
def clean_num(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"[^\d.-]","",str(x))
    return s if s else np.nan

for c in ["Age","WBC_Count","Heart_Rate","Platelet_Count","Tumor Size"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].apply(clean_num), errors="coerce")

# ---------------- AGE RULE ----------------
if "Age" in df.columns:
    df.loc[(df["Age"]<0)|(df["Age"]>120),"Age"] = np.nan

# ---------------- BLOOD GROUP ----------------
valid_bg = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]
df.loc[~df["Blood_Group"].isin(valid_bg),"Blood_Group"] = np.nan

# ---------------- BLOOD PRESSURE ----------------
if "Blood_Pressure" in df.columns:
    df["Blood_Pressure"] = df["Blood_Pressure"].apply(lambda x: x if re.match(r"^\d{2,3}/\d{2,3}$",str(x)) else np.nan)

# ---------------- DOB CLEAN ----------------
if "DOB" in df.columns:
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce", dayfirst=True)
    df["Age"] = pd.Timestamp.today().year - df["DOB"].dt.year

# ---------------- DIAGNOSIS LOGIC ----------------
if "Diagnosis_Status" in df.columns:
    neg = df["Diagnosis_Status"].str.lower()=="negative"
    if "Cancer_Type" in df.columns:
        df.loc[neg,"Cancer_Type"] = "NA"
    if "Stages" in df.columns:
        df.loc[neg,"Stages"] = "NA"
    if "Tumor Size" in df.columns:
        df.loc[neg,"Tumor Size"] = 0

# ---------------- OUTLIERS ----------------
df.loc[df["WBC_Count"]>50000,"WBC_Count"] = np.nan
df.loc[df["Platelet_Count"]>900,"Platelet_Count"] = np.nan
df.loc[df["Heart_Rate"]>220,"Heart_Rate"] = np.nan
df.loc[(df["Tumor Size"]<0)|(df["Tumor Size"]>100),"Tumor Size"] = np.nan

# ---------------- IMPUTE ----------------
for c in df.select_dtypes(include=np.number):
    df[c] = df[c].fillna(df[c].median())

for c in df.select_dtypes(include="object"):
    df[c] = df[c].fillna("NA")

# ---------------- FINAL TYPES ----------------
if "Age" in df.columns:
    df["Age"] = df["Age"].round().astype(int)
if "Heart_Rate" in df.columns:
    df["Heart_Rate"] = df["Heart_Rate"].round().astype(int)
if "WBC_Count" in df.columns:
    df["WBC_Count"] = df["WBC_Count"].round(1)

# -------- FINAL PRESENTATION PATCH --------

# 1. DOB → format DD/MM/YYYY (remove timestamps)
if "DOB" in df.columns:
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce", dayfirst=True)
    df["DOB"] = df["DOB"].dt.strftime("%d/%m/%Y")

# 2. Alcoholic_Consumption & Tobacco_Usage: replace NA with "None"
for col in ["Acholic_Consumption", "Tobacco_Usage"]:
    if col in df.columns:
        df[col] = df[col].replace("NA", "None")

# 3. Tumor Size: convert 0 → "NA" for Negative patients
if "Diagnosis_Status" in df.columns and "Tumor Size" in df.columns:
    neg = df["Diagnosis_Status"].str.lower() == "negative"
    df.loc[neg, "Tumor Size"] = "NA"

os.makedirs(output_folder, exist_ok=True)
df.to_excel(output_file, index=False)

print("CLEAN DATASET SAVED SUCCESSFULLY")
print(f"Cleaned dataset saved to: {output_file}")