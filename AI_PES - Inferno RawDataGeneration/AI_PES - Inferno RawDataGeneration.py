# Import necessary libraries
import pandas as pd
import numpy as np
from faker import Faker
from datetime import date

# Initialize Faker for generating realistic data
faker = Faker()
today = date.today()
# Number of patients to generate
num_patients = 1000

# Initialize lists to store generated data
patient_ids = []
full_names = []
DOB = []
ages = []
genders = []
blood_groups = []
physical_activities = []
acholic_consumptions = []
tobacco_usages = []
wbc_counts = []
rbc_counts = []
platelet_counts = []
hemoglobin_levels = []
blood_pressures = []
heart_rates = []
diagnosis_statuses = [] # Corrected: Renamed and initialized as an empty list
stages = []
tumor_sizes = []
cancer_types = []

# Generate data for each patient
for i in range(num_patients):
    # Patient_ID: Unique identifier
    patient_ids.append(f"PID{i+1:04d}")
    
    # Full_Name: Generated using Faker
    full_names.append(faker.name())
    
    # DOB: Generated using Faker
    birth_date = faker.date_of_birth(minimum_age=2, maximum_age=90)
    birth_str = birth_date.strftime("%d/%m/%Y")
    DOB.append(birth_str)

    # Age: Random integer within a realistic range
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    ages.append(age)
    
    # Gender: Randomly selected
    gender = np.random.choice(["Male", "Female"])
    genders.append(gender)
    
    # Blood_Group: Randomly selected
    blood_groups.append(np.random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]))
    
    # Physical_Activity: Randomly selected
    physical_activities.append(np.random.choice(["Low", "Moderate", "High"]))

    # Acholic_Consumption: Randomly selected
    acholic_consumptions.append(np.random.choice(["None", "Occasional", "Regular"]))

    # Tobacco_Usage: Randomly selected
    tobacco_usages.append(np.random.choice(["None", "Occasional", "Regular"]))

    # WBC_Count: Normally distributed within a realistic range
    wbc_counts.append(round(np.random.uniform(4.0, 11.0), 1))
    
    # RBC_Count: Normally distributed within a realistic range
    rbc_counts.append(round(np.random.uniform(4.0, 5.5), 1))
    
    # Platelet_Count: Normally distributed within a realistic range
    platelet_counts.append(np.random.randint(150, 451))
    
    # Hemoglobin_Level: Normally distributed within a realistic range
    hemoglobin_levels.append(round(np.random.uniform(12.0, 17.5), 1))
    
    # Blood_Pressure: Systolic/Diastolic format within realistic ranges
    systolic = np.random.randint(100, 140)
    diastolic = np.random.randint(60, 90)
    blood_pressures.append(f"{systolic}/{diastolic}")
    
    # Heart_Rate: Random integer within a realistic range
    heart_rates.append(np.random.randint(60, 101))
    
    current_diagnosis_status = np.random.choice(["Positive", "Negative"], p=[0.5,0.5]) # Temporary variable
    diagnosis_statuses.append(current_diagnosis_status) # Append to the list
    
    if current_diagnosis_status == "Positive":
        cancer_types.append(np.random.choice(["Malignant", "Benign"]))
        stages.append(np.random.choice(["I", "II", "III", "IV"]))
        tumor_sizes.append(round(np.random.uniform(1.0, 12.0), 1)) # cm
    else:
        # If No Cancer, these fields MUST be Empty (None/NaN)
        cancer_types.append("N/A") 
        stages.append("N/A")
        tumor_sizes.append("N/A") 

# Create a dictionary from the generated data
data = {
    "Patient_ID": patient_ids,
    "Full_Name": full_names,
    "DOB": DOB,
    "Age": ages,
    "Gender": genders,
    "Blood_Group": blood_groups,
    "Physical_Activity": physical_activities,
    "Acholic_Consumption": acholic_consumptions,
    "Tobacco_Usage": tobacco_usages,
    "WBC_Count": wbc_counts,
    "RBC_Count": rbc_counts,
    "Platelet_Count": platelet_counts,
    "Hemoglobin_Level": hemoglobin_levels,
    "Blood_Pressure": blood_pressures,
    "Heart_Rate": heart_rates,
    "Diagnosis_Status": diagnosis_statuses,
    "Cancer_Type": cancer_types,
    "Stages": stages,
    "Tumor Size": tumor_sizes
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
excel_file_path = "AI_PES - Inferno RawDataGeneration/AI_PES - Inferno RawDataset.xlsx"
df.to_excel(excel_file_path, index=False)
print(f"Raw dataset generated and saved to {excel_file_path}")
