import pandas as pd
from sklearn.model_selection import train_test_split

# GitHub RAW Excel file
feature_file_path = (
    "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20FeatureEngineering/AI_PES%20-%20Inferno%20Feature_Engineered_Data.xlsx"
)

# Read Excel file from GitHub
data = pd.read_excel(feature_file_path,engine='openpyxl')

# Clean column names (remove spaces, lowercase)
data.columns = data.columns.str.strip()

print("Columns in dataset:", list(data.columns))

# Automatically detect target column
possible_labels = ["Diagnosis", "diagnosis", "Class", "class", "target", "Label"]

label_column = None
for col in possible_labels:
    if col in data.columns:
        label_column = col
        break

# Fallback: use last column if not found
if label_column is None:
    label_column = data.columns[-1]

print("Using label column:", label_column)

# Split features and target
X = data.drop(columns=[label_column])
y = data[label_column]

# Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Combine features and labels
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save output files
train_data.to_excel("AI_PES - Inferno SplittingofData/AI_PES - Inferno Training_data.xlsx", index=False)
test_data.to_excel("AI_PES - Inferno SplittingofData/AI_PES - Inferno Testing_data.xlsx", index=False)

print("✅ Data splitting completed successfully!")
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)