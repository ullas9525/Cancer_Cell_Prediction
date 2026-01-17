import pandas as pd
import numpy as np
import os
from faker import Faker
from datetime import date

# Initialize Faker
faker = Faker()
today = date.today()

# CONFIG
num_patients = 1000
output_file = "AI_PES - Inferno RawDataGeneration/AI_PES - Inferno RawDataset.xlsx"

# Lists to store data
patient_ids = []
full_names = []
dobs = []
ages = []
genders = []
blood_groups = []
family_histories = []
genetic_markers = []
bmis = []
physical_activities = []
acholic_consumptions = []
tobacco_usages = []
wbc_counts = []
rbc_counts = []
platelet_counts = []
hemoglobin_levels = []
blood_pressures = []
heart_rates = []
diagnosis_statuses = []
cancer_types = []
stages = []
tumor_sizes = []

print(f"Generating {num_patients} patients with Risk Logic...")

for i in range(num_patients):
    # --- 1. Basic Info ---
    # Age (Weighted towards older for cancer realism)
    age = int(np.random.triangular(18, 50, 90))
    ages.append(age)
    
    birth_year = today.year - age
    dob = faker.date_of_birth(minimum_age=age, maximum_age=age).strftime("%d/%m/%Y")
    dobs.append(dob)
    
    gender = np.random.choice(["Male", "Female"])
    genders.append(gender)
    
    name = faker.name_male() if gender == "Male" else faker.name_female()
    full_names.append(name)
    
    pid = f"PID{i+1:04d}"
    patient_ids.append(pid)
    
    blood_groups.append(np.random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]))
    
    # --- 2. Risk Factors (Independent Variables) ---
    # Genetics
    family_history = np.random.choice(["Yes", "No"], p=[0.15, 0.85])
    family_histories.append(family_history)
    
    genetic_marker = np.random.choice(["Present", "Absent"], p=[0.05, 0.95]) # Rare but critical
    genetic_markers.append(genetic_marker)
    
    # Body Metrics
    height_m = np.random.normal(1.7, 0.1)
    weight_kg = np.random.normal(70, 15)
    bmi = round(weight_kg / (height_m ** 2), 1)
    bmis.append(bmi)
    
    # Lifestyle
    tobacco = np.random.choice(["None", "Occasional", "Regular"], p=[0.6, 0.2, 0.2])
    tobacco_usages.append(tobacco)
    
    alcohol = np.random.choice(["None", "Occasional", "Regular"], p=[0.5, 0.3, 0.2])
    acholic_consumptions.append(alcohol)
    
    activity = np.random.choice(["Low", "Moderate", "High"], p=[0.3, 0.4, 0.3])
    physical_activities.append(activity)
    
    # Vitals
    wbc = round(np.random.normal(7.0, 2.0), 1) # Normal 4.5-11. Cancer often higher.
    wbc_counts.append(wbc)
    
    rbc = round(np.random.normal(4.8, 0.5), 1)
    rbc_counts.append(rbc)
    
    platelets = int(np.random.normal(250, 50))
    platelet_counts.append(platelets)
    
    hb = round(np.random.normal(14, 1.5), 1)
    hemoglobin_levels.append(hb)
    
    systolic = int(np.random.normal(120, 15))
    diastolic = int(np.random.normal(80, 10))
    blood_pressures.append(f"{systolic}/{diastolic}")
    
    hr = int(np.random.normal(75, 10))
    heart_rates.append(hr)
    
    # --- 3. RISK SCORE CALCULATION (The Logic Core) ---
    risk_score = 0
    
    # Age Factor
    if age > 50: risk_score += (age - 50) * 0.5
    
    # Genetics (Strong Determinants) - BOOSTED WEIGHTS
    if genetic_marker == "Present": risk_score += 80 # Was 60
    if family_history == "Yes": risk_score += 40 # Was 25
    
    # Lifestyle Impact
    if tobacco == "Regular": risk_score += 35 # Was 25
    elif tobacco == "Occasional": risk_score += 15
    
    if alcohol == "Regular": risk_score += 20
    
    if activity == "Low": risk_score += 15
    elif activity == "High": risk_score -= 15 # protective
    
    # BMI Impact
    if bmi > 30: risk_score += 20 # Obese
    elif bmi > 25: risk_score += 10 # Overweight
    
    # Vitals Impact (Symptoms)
    if wbc > 11.0: risk_score += 25 
    if wbc < 4.0: risk_score += 15
    
    # --- 4. Diagnosis Assignment ---
    # Sigmoid-ish probability map
    # Base risk is low (~10%). Max risk approaches 95%.
    # threshold normalization
    # Map score 0 -> 5% prob, Score 100 -> 95% prob
    # STEEPER CURVE: Divisor changed from 15 to 8. This makes the boundary sharper.
    prob_positive = 1 / (1 + np.exp(-(risk_score - 50) / 8)) 
    
    diagnosis = np.random.choice(["Positive", "Negative"], p=[prob_positive, 1 - prob_positive])
    diagnosis_statuses.append(diagnosis)
    
    # --- 5. Post-Diagnosis Features (Dependent on Result) ---
    c_type = "N/A"
    stg = "N/A"
    t_size = "N/A"
    
    if diagnosis == "Positive":
        c_type = np.random.choice(["Malignant", "Benign"], p=[0.7, 0.3])
        # Higher risk score -> Higher likelihood of advanced stage?
        stage_probs = [0.4, 0.3, 0.2, 0.1]
        if risk_score > 60: stage_probs = [0.1, 0.2, 0.3, 0.4]
        stg = np.random.choice(["I", "II", "III", "IV"], p=stage_probs)
        
        # Tumor Size (cm)
        base_size = np.random.lognormal(1.0, 0.5) # mostly small, some distinct
        if stg == "III": base_size += 2
        if stg == "IV": base_size += 4
        t_size = round(base_size, 1)
        
    cancer_types.append(c_type)
    stages.append(stg)
    tumor_sizes.append(t_size)

# Create DataFrame
data = {
    "Patient_ID": patient_ids,
    "Full_Name": full_names,
    "DOB": dobs,
    "Age": ages,
    "Gender": genders,
    "Blood_Group": blood_groups,
    "Family_Cancer_History": family_histories,
    "Genetic_Markers": genetic_markers,
    "BMI": bmis,
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

df = pd.DataFrame(data)

# Save
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_excel(output_file, index=False)
print(f"Dataset generated: {output_file}")