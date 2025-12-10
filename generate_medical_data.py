# Import necessary libraries
import pandas as pd
import numpy as np
from faker import Faker
import random

# Set a fixed random seed for reproducibility
np.random.seed(42)
Faker.seed(42)
random.seed(42)

# Initialize Faker
fake = Faker()

# Number of patients
num_patients = 1000

# Generate Patient_ID
patient_ids = [f"PID{i+1:04d}" for i in range(num_patients)]

# Generate Name, Age, Gender
names = [fake.name() for _ in range(num_patients)]
ages = np.random.randint(20, 80, num_patients)
genders = np.random.choice(["Male", "Female"], num_patients)

# Generate WBC_Count (White Blood Cell Count) - higher for cancer patients
wbc_counts = np.random.normal(loc=7000, scale=1500, size=num_patients).astype(int)

# Generate Tumor_Size (in mm) - present for cancer patients
tumor_sizes = np.random.normal(loc=25, scale=10, size=num_patients)
tumor_sizes[tumor_sizes < 0] = 0 # Ensure non-negative tumor size initially

# Generate Diagnosis (Cancer or No Cancer)
diagnosis = np.random.choice(["Cancer", "No Cancer"], num_patients, p=[0.3, 0.7])

# Adjust WBC_Count and Tumor_Size based on diagnosis
for i in range(num_patients):
    if diagnosis[i] == "Cancer":
        wbc_counts[i] = np.random.normal(loc=12000, scale=3000) # Higher WBC for cancer
        tumor_sizes[i] = np.random.normal(loc=40, scale=15)    # Larger tumor for cancer
        if tumor_sizes[i] < 0: tumor_sizes[i] = 0
    else:
        tumor_sizes[i] = 0 # No tumor for \'No Cancer\'

wbc_counts = wbc_counts.astype(int)

# Generate Stage (for Cancer patients) and Treatment_Recommended
stages = []
treatments = []

for d in diagnosis:
    if d == "Cancer":
        stage = np.random.choice(["Stage I", "Stage II", "Stage III", "Stage IV"], p=[0.3, 0.3, 0.25, 0.15])
        treatment = np.random.choice(["Chemotherapy", "Radiation", "Surgery", "Immunotherapy"])
    else:
        stage = "N/A"
        treatment = "Observation"
    stages.append(stage)
    treatments.append(treatment)

# Create initial DataFrame
data = pd.DataFrame({
    "Patient_ID": patient_ids,
    "Name": names,
    "Age": ages,
    "Gender": genders,
    "WBC_Count": wbc_counts,
    "Tumor_Size": tumor_sizes.round(2),
    "Diagnosis": diagnosis,
    "Stage": stages,
    "Treatment": treatments # Changed from Treatment_Recommended to Treatment as per task
})

# Convert Age and WBC_Count to object dtype to allow mixed types without warning
data["Age"] = data["Age"].astype(object)
data["WBC_Count"] = data["WBC_Count"].astype(object)

# --- Introduce Real-World Anomalies ---

# Number of records to inject anomalies into (approx 30%)
num_anomalies_records = int(num_patients * 0.30)
anomaly_indices = np.random.choice(num_patients, num_anomalies_records, replace=False)

# 1. Extreme ages (negative values, age >120)
extreme_age_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.2), replace=False)
for idx in extreme_age_indices:
    if np.random.rand() < 0.5:
        data.loc[idx, "Age"] = np.random.randint(-10, 0)  # Negative age
    else:
        data.loc[idx, "Age"] = np.random.randint(121, 150) # Age > 120

# 2. Inconsistent Gender values (M, Female, femlae, Unknown)
wrong_gender_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.3), replace=False)
wrong_genders = ["M", "Female", "femlae", "Unknown"]
for idx in wrong_gender_indices:
    data.loc[idx, "Gender"] = random.choice(wrong_genders)

# 3. Random missing values in Age, Stage, and WBC_Count
null_age_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.2), replace=False)
data.loc[null_age_indices, "Age"] = np.nan
null_stage_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.2), replace=False)
data.loc[null_stage_indices, "Stage"] = np.nan
null_wbc_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.2), replace=False)
data.loc[null_wbc_indices, "WBC_Count"] = np.nan

# 4. Extreme outliers (Tumor_Size > 100)
tumor_outlier_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.15), replace=False)
for idx in tumor_outlier_indices:
    data.loc[idx, "Tumor_Size"] = round(np.random.uniform(101, 200), 2) # Very large tumor size

# 5. Typos in Stage (e.g., "staeg 3")
typo_stage_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.2), replace=False)
typo_stages = {"Stage I": "Staeg 1", "Stage II": "staeg 2", "Stage III": "Stag 3", "Stage IV": "stage IV"}
for idx in typo_stage_indices:
    if data.loc[idx, "Stage"] in typo_stages:
        data.loc[idx, "Stage"] = typo_stages[data.loc[idx, "Stage"]]
    elif data.loc[idx, "Stage"] == "N/A" and np.random.rand() < 0.3: # Introduce typos for N/A sometimes too
        data.loc[idx, "Stage"] = "n/a"

# 6. Mixed label formats (Cancer, cancer, CANCER, No cancer)
diagnosis_inconsistency_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.3), replace=False)
inconsistent_diagnosis_labels = ["cancer", "CANCER", "No cancer"]
for idx in diagnosis_inconsistency_indices:
    if data.loc[idx, "Diagnosis"] == "Cancer":
        data.loc[idx, "Diagnosis"] = random.choice(["cancer", "CANCER"])
    elif data.loc[idx, "Diagnosis"] == "No Cancer":
        data.loc[idx, "Diagnosis"] = "No cancer"

# 7. Wrong data types (Age as string, WBC_Count as text)
wrong_dtype_age_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.1), replace=False)
for idx in wrong_dtype_age_indices:
    data.loc[idx, "Age"] = str(data.loc[idx, "Age"]) # Convert existing age to string

wrong_dtype_wbc_indices = np.random.choice(anomaly_indices, int(num_anomalies_records * 0.1), replace=False)
for idx in wrong_dtype_wbc_indices:
    data.loc[idx, "WBC_Count"] = fake.word() # Random string

# 8. Duplicate Patient_ID for 3 random patients
duplicate_id_indices = np.random.choice(num_patients, 3, replace=False)
if num_patients > 1:
    # Make 3 patients share the same Patient_ID
    shared_id = data.loc[duplicate_id_indices[0], "Patient_ID"]
    data.loc[duplicate_id_indices[1], "Patient_ID"] = shared_id
    data.loc[duplicate_id_indices[2], "Patient_ID"] = shared_id

# Export the raw dataset with anomalies
data.to_csv("synthetic_raw.csv", index=False)
print("Synthetic medical dataset generated and saved to synthetic_raw.csv Successfully.")
