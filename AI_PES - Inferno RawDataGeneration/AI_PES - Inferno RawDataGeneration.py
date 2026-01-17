import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import os  # Import os for file path handling
from faker import Faker  # Import Faker to generate realistic fake data
from datetime import date  # Import date to calculate ages

# Initialize Faker
faker = Faker()  # Create Faker instance
today = date.today()  # Get current date

# CONFIG
num_patients = 1000  # Set number of patients to generate
output_file = "AI_PES - Inferno RawDataGeneration/AI_PES - Inferno RawDataset.xlsx"  # Output file path

# Lists to store data
patient_ids = []  # List to hold Patient IDs
full_names = []  # List to hold Full Names
dobs = []  # List to hold Dates of Birth
ages = []  # List to hold Ages
genders = []  # List to hold Genders
blood_groups = []  # List to hold Blood Groups
family_histories = []  # List for Family History status
genetic_markers = []  # List for Genetic Markers status
bmis = []  # List for BMI values
physical_activities = []  # List for Physical Activity levels
acholic_consumptions = []  # List for Alcohol Consumption habits
tobacco_usages = []  # List for Tobacco Usage habits
wbc_counts = []  # List for White Blood Cell counts
rbc_counts = []  # List for Red Blood Cell counts
platelet_counts = []  # List for Platelet counts
hemoglobin_levels = []  # List for Hemoglobin levels
blood_pressures = []  # List for Blood Pressure readings
heart_rates = []  # List for Heart Rate readings
diagnosis_statuses = []  # List for Final Diagnosis (Target)
cancer_types = []  # List for Cancer Type (if Positive)
stages = []  # List for Cancer Stage (if Positive)
tumor_sizes = []  # List for Tumor Size (if Positive)

print(f"Generating {num_patients} patients with Risk Logic...")  # Notify user of start

for i in range(num_patients):  # Loop through sample count
    # --- 1. Basic Info ---
    # Age (Weighted towards older for cancer realism)
    age = int(np.random.triangular(18, 50, 90))  # Generate realistic skewed age
    ages.append(age)  # Store age
    
    birth_year = today.year - age  # Calculate birth year
    dob = faker.date_of_birth(minimum_age=age, maximum_age=age).strftime("%d/%m/%Y")  # Generate DOB
    dobs.append(dob)  # Store DOB
    
    gender = np.random.choice(["Male", "Female"])  # Randomly select gender
    genders.append(gender)  # Store gender
    
    name = faker.name_male() if gender == "Male" else faker.name_female()  # Generate appropriate name
    full_names.append(name)  # Store name
    
    pid = f"PID{i+1:04d}"  # Create unique ID
    patient_ids.append(pid)  # Store ID
    
    blood_groups.append(np.random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]))  # Random Blood Group
    
    # --- 2. Risk Factors (Independent Variables) ---
    # Genetics
    family_history = np.random.choice(["Yes", "No"], p=[0.15, 0.85])  # Weighted random choice for family history
    family_histories.append(family_history)  # Store history
    
    genetic_marker = np.random.choice(["Present", "Absent"], p=[0.05, 0.95]) # Rare but critical marker
    genetic_markers.append(genetic_marker)  # Store marker
    
    # Body Metrics
    height_m = np.random.normal(1.7, 0.1)  # Normal distribution for height
    weight_kg = np.random.normal(70, 15)  # Normal distribution for weight
    bmi = round(weight_kg / (height_m ** 2), 1)  # Calculate body mass index
    bmis.append(bmi)  # Store BMI
    
    # Lifestyle
    tobacco = np.random.choice(["None", "Occasional", "Regular"], p=[0.6, 0.2, 0.2])  # Generate tobacco habit
    tobacco_usages.append(tobacco)  # Store tobacco usage
    
    alcohol = np.random.choice(["None", "Occasional", "Regular"], p=[0.5, 0.3, 0.2])  # Generate alcohol habit
    acholic_consumptions.append(alcohol)  # Store alcohol usage
    
    activity = np.random.choice(["Low", "Moderate", "High"], p=[0.3, 0.4, 0.3])  # Generate activity level
    physical_activities.append(activity)  # Store activity
    
    # Vitals
    wbc = round(np.random.normal(7.0, 2.0), 1) # Normal 4.5-11. Cancer often higher.
    wbc_counts.append(wbc)  # Store WBC count
    
    rbc = round(np.random.normal(4.8, 0.5), 1)  # Generate RBC count
    rbc_counts.append(rbc)  # Store RBC count
    
    platelets = int(np.random.normal(250, 50))  # Generate Platelet count
    platelet_counts.append(platelets)  # Store Platelet count
    
    hb = round(np.random.normal(14, 1.5), 1)  # Generate Hemoglobin level
    hemoglobin_levels.append(hb)  # Store Hemoglobin
    
    systolic = int(np.random.normal(120, 15))  # Generate Systolic BP
    diastolic = int(np.random.normal(80, 10))  # Generate Diastolic BP
    blood_pressures.append(f"{systolic}/{diastolic}")  # Format and store BP
    
    hr = int(np.random.normal(75, 10))  # Generate Heart Rate
    heart_rates.append(hr)  # Store Heart Rate
    
    # --- 3. RISK SCORE CALCULATION (The Logic Core) ---
    risk_score = 0  # Initialize risk score
    
    # Age Factor
    if age > 50: risk_score += (age - 50) * 0.5  # Add risk for advanced age
    
    # Genetics (Strong Determinants) - BOOSTED WEIGHTS
    if genetic_marker == "Present": risk_score += 80 # High penalty for genetics
    if family_history == "Yes": risk_score += 40 # Moderate penalty for family history
    
    # Lifestyle Impact
    if tobacco == "Regular": risk_score += 35 # High penalty for smoking
    elif tobacco == "Occasional": risk_score += 15  # Low penalty for occasional smoking
    
    if alcohol == "Regular": risk_score += 20  # Penalty for drinking
    
    if activity == "Low": risk_score += 15  # Penalty for low activity
    elif activity == "High": risk_score -= 15 # Benefit (protective) for high activity
    
    # BMI Impact
    if bmi > 30: risk_score += 20 # Penalty for Obesity
    elif bmi > 25: risk_score += 10 # Penalty for Overweight
    
    # Vitals Impact (Symptoms)
    if wbc > 11.0: risk_score += 25  # Risk indicator (Infection/Leukemia sign)
    if wbc < 4.0: risk_score += 15  # Risk indicator (Compromised immunity)
    
    # --- 4. Diagnosis Assignment ---
    # Sigmoid-ish probability map
    # Base risk is low (~10%). Max risk approaches 95%.
    # threshold normalization
    # Map score 0 -> 5% prob, Score 100 -> 95% prob
    # STEEPER CURVE: Divisor changed from 15 to 8. This makes the boundary sharper.
    prob_positive = 1 / (1 + np.exp(-(risk_score - 50) / 8))  # Calculate cancer probability using sigmoid
    
    diagnosis = np.random.choice(["Positive", "Negative"], p=[prob_positive, 1 - prob_positive])  # Assign diagnosis based on prob
    diagnosis_statuses.append(diagnosis)  # Store diagnosis
    
    # --- 5. Post-Diagnosis Features (Dependent on Result) ---
    c_type = "N/A"  # Default to N/A
    stg = "N/A"  # Default to N/A
    t_size = "N/A"  # Default to N/A
    
    if diagnosis == "Positive":
        c_type = np.random.choice(["Malignant", "Benign"], p=[0.7, 0.3])  # Assign type if positive
        # Higher risk score -> Higher likelihood of advanced stage?
        stage_probs = [0.4, 0.3, 0.2, 0.1]
        if risk_score > 60: stage_probs = [0.1, 0.2, 0.3, 0.4]  # Weighted stages for high risk
        stg = np.random.choice(["I", "II", "III", "IV"], p=stage_probs)  # Assign stage
        
        # Tumor Size (cm)
        base_size = np.random.lognormal(1.0, 0.5) # mostly small, some distinct
        if stg == "III": base_size += 2  # Larger size for stage III
        if stg == "IV": base_size += 4  # Largest size for stage IV
        t_size = round(base_size, 1)  # Round tumor size
        
    cancer_types.append(c_type)  # Store type
    stages.append(stg)  # Store stage
    tumor_sizes.append(t_size)  # Store size

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

df = pd.DataFrame(data)  # Convert dictionary to DataFrame

# Save
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Create directory if missing
df.to_excel(output_file, index=False)  # Save DataFrame to Excel
print(f"Dataset generated: {output_file}")  # Confirm save completion