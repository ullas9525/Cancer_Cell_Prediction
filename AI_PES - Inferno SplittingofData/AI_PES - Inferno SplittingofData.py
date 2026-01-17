import pandas as pd  # Import pandas for data manipulation
from sklearn.model_selection import train_test_split  # Import function for data splitting

# GitHub RAW Excel file
feature_file_path = (
    "https://raw.githubusercontent.com/ullas9525/Cancer_Cell_Prediction/main/AI_PES%20-%20Inferno%20FeatureEngineering/AI_PES%20-%20Inferno%20Feature_Engineered_Data.xlsx"  # Input file path
)

# Read Excel file from GitHub
data = pd.read_excel(feature_file_path, engine='openpyxl')  # Load data into DataFrame

# Clean column names (remove spaces, lowercase)
data.columns = data.columns.str.strip()  # Remove whitespace from column headers

print("Columns in dataset:", list(data.columns))  # Print column names for verification

# Automatically detect target column
possible_labels = ["Diagnosis", "diagnosis", "Class", "class", "target", "Label"]  # List of likely target names

label_column = None  # Initialize variable
for col in possible_labels:  # Check each candidate
    if col in data.columns:  # If found in dataset
        label_column = col  # Set as label
        break  # Stop searching

# Fallback: use last column if not found
if label_column is None:  # If no match found
    label_column = data.columns[-1]  # Default to last column

print("Using label column:", label_column)  # Confirm target column

# Split features and target
X = data.drop(columns=[label_column])  # Define features
y = data[label_column]  # Define target

# Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Features
    y,  # Target
    test_size=0.2,  # 20% for testing
    random_state=42,  # Seed for reproducibility
    stratify=y  # Maintain class balance
)

# Combine features and labels
train_data = pd.concat([X_train, y_train], axis=1)  # Reassemble train set
test_data = pd.concat([X_test, y_test], axis=1)  # Reassemble test set

# Save output files
train_data.to_excel("AI_PES - Inferno SplittingofData/AI_PES - Inferno Training_data.xlsx", index=False)  # Save training data
test_data.to_excel("AI_PES - Inferno SplittingofData/AI_PES - Inferno Testing_data.xlsx", index=False)  # Save testing data

print("✅ Data splitting completed successfully!")  # Success message
print("Train data shape:", train_data.shape)  # Print train dimensions
print("Test data shape:", test_data.shape)  # Print test dimensions