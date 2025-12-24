import pandas as pd                     # load pandas
import numpy as np                      # load numpy
import random                           # load random

RANDOM_SEED = 42                        # set random seed
random.seed(RANDOM_SEED)                # apply seed to random
np.random.seed(RANDOM_SEED)             # apply seed to numpy

url = "AI_PES - Inferno RawDataGeneration/AI_PES - Inferno RawDataset.xlsx"  # dataset path

try:                                    # try loading file
    df = pd.read_excel(url)             # read excel
    print("Dataset loaded successfully.") # success message
except Exception as e:                  # catch error
    print(f"Error loading dataset: {e}") # print error
    exit()                              # stop execution

df_anomalous = df.copy()                # copy dataset
n_rows = len(df_anomalous)              # count rows

COLUMNS = {                             # define columns
    "Age": "numeric",
    "Blood_Group": "categorical",
    "WBC_Count": "numeric",
    "Blood_Pressure": "string",
    "Heart_Rate": "numeric",
    "Gender": "categorical",
    "Physical_Activity": "categorical",
    "Alcohol_Consumption": "categorical",
    "Tobacco_Usage": "categorical",
    "Platelet_Count": "numeric",
    "Cancer_Type": "categorical",
    "Patient_ID": "string"
}

def inject_null_missing_data(df, columns_to_affect, rate_per_column):  # missing data injector
    for col in columns_to_affect:                                      # loop columns
        num_to_affect = int(len(df) * rate_per_column)                 # count rows
        indices = np.random.choice(df.index, size=num_to_affect, replace=False) # select rows
        df.loc[indices, col] = np.nan                                  # insert NaN
    return df                                                          # return df

def inject_column_misplacement(df, misplacement_map, rate):            # misplacement injector
    num_to_affect = int(len(df) * rate)                                # count rows
    indices = np.random.choice(df.index, size=num_to_affect, replace=False) # select rows
    for target_col, source_col in misplacement_map.items():            # loop mapping
        if target_col in df.columns and source_col in df.columns:      # validate columns
            temp_values = df.loc[indices, source_col].copy()           # copy values
            df.loc[indices, target_col] = temp_values                  # place wrongly
    return df                                                          # return df

def inject_wrong_data_types(df, type_conversion_map, rate):            # datatype injector
    num_to_affect = int(len(df) * rate)                                # count rows
    indices = np.random.choice(df.index, size=num_to_affect, replace=False) # select rows
    for col, conversion_type in type_conversion_map.items():           # loop types
        if col in df.columns:                                          # validate column
            if conversion_type == "numeric_to_text":                   # numeric to text
                text_options = {
                    "Age": ["Twenty", "Forty", "Sixty", "Thirty-Five", "Old"],
                    "WBC_Count": ["Normal", "High", "Low", "Elevated", "Suppressed"],
                    "Heart_Rate": ["Fast", "Slow", "Average", "Rapid"]
                }
                df.loc[indices, col] = [random.choice(text_options.get(col, ["text_val"])) for _ in range(num_to_affect)] # convert
            elif conversion_type == "text_to_numeric":                 # text to numeric
                if col == "Gender":
                    df.loc[indices, col] = np.random.choice([0, 1], size=num_to_affect) # convert
    return df                                                          # return df

def inject_out_of_range_values(df, out_of_range_map, rate):             # range injector
    num_to_affect = int(len(df) * rate)                                # count rows
    indices = np.random.choice(df.index, size=num_to_affect, replace=False) # select rows
    for col, ranges in out_of_range_map.items():                       # loop ranges
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]): # validate numeric
            out_values = np.random.choice(ranges, size=num_to_affect)  # choose values
            df.loc[indices, col] = out_values                          # insert values
    return df                                                          # return df

def inject_invalid_categorical_values(df, categorical_map, rate):      # categorical injector
    num_to_affect = int(len(df) * rate)                                # count rows
    indices = np.random.choice(df.index, size=num_to_affect, replace=False) # select rows
    for col, invalid_options in categorical_map.items():               # loop categories
        if col in df.columns:
            df.loc[indices, col] = [random.choice(invalid_options) for _ in range(num_to_affect)] # insert
    return df                                                          # return df

def inject_format_corruption(df, format_corruption_map, rate):         # format injector
    num_to_affect = int(len(df) * rate)                                # count rows
    indices = np.random.choice(df.index, size=num_to_affect, replace=False) # select rows
    for col, corruption_type in format_corruption_map.items():         # loop corruption
        if col in df.columns:
            if corruption_type == "blood_pressure_malformed":
                malformed_bp = ["120", "/80", "120/", "High BP", "abc/xyz", "120-80"]
                df.loc[indices, col] = [random.choice(malformed_bp) for _ in range(num_to_affect)]
            elif corruption_type == "leading_trailing_spaces":
                df.loc[indices, col] = df.loc[indices, col].astype(str).apply(lambda x: f" {x} ")
            elif corruption_type == "mixed_casing":
                df.loc[indices, col] = df.loc[indices, col].astype(str).apply(lambda x: x.upper() if random.random() < 0.5 else x.lower())
    return df                                                          # return df

def inject_duplicate_records(df, num_duplicates):                      # duplicate injector
    if num_duplicates > 0:
        duplicate_rows = df.sample(n=num_duplicates, replace=True)     # sample rows
        df = pd.concat([df, duplicate_rows], ignore_index=True)        # append rows
    return df                                                          # return df

null_cols = ["Age", "Blood_Group", "WBC_Count", "Blood_Pressure", "Heart_Rate"]  # null columns
df_anomalous = inject_null_missing_data(df_anomalous, null_cols, 0.05) # inject null

misplacement_map = {
    "Gender": "Age",
    "Age": "Gender",
    "Blood_Group": "Gender",
    "Physical_Activity": random.choice(["Alcohol_Consumption", "Tobacco_Usage"]),
    "Tobacco_Usage": "Physical_Activity"
}
df_anomalous = inject_column_misplacement(df_anomalous, misplacement_map, 0.04) # misplace

type_conversion_map = {
    "Age": "numeric_to_text",
    "WBC_Count": "numeric_to_text",
    "Heart_Rate": "numeric_to_text",
    "Gender": "text_to_numeric"
}
df_anomalous = inject_wrong_data_types(df_anomalous, type_conversion_map, 0.03) # wrong types

out_of_range_map = {
    "Age": [-5, -1, 130, 150],
    "WBC_Count": [50, 500000, 999999, 10],
    "Platelet_Count": [10, 1200, 1500000, 50],
    "Heart_Rate": [20, 250, 30, 280]
}
df_anomalous = inject_out_of_range_values(df_anomalous, out_of_range_map, 0.04) # out range

categorical_map = {
    "Gender": ["M", "FEMALEEE", "UNKNOWN", "not specified", "F "],
    "Blood_Group": ["A++", "O ", "C-", "??", "B+", "AB-ve"],
    "Cancer_Type": ["breast cancer", "lung-cancer", "UNKNOWN", "type 1", "CANCER_TYPE"]
}
df_anomalous = inject_invalid_categorical_values(df_anomalous, categorical_map, 0.05) # invalid

format_corruption_map = {
    "Blood_Pressure": "blood_pressure_malformed",
    "Gender": "leading_trailing_spaces",
    "Blood_Group": "mixed_casing",
    "Cancer_Type": "leading_trailing_spaces"
}
df_anomalous = inject_format_corruption(df_anomalous, format_corruption_map, 0.04) # corrupt

num_duplicates_to_add = int(n_rows * 0.02)        # duplicate count
df_anomalous = inject_duplicate_records(df_anomalous, num_duplicates_to_add) # duplicate

print("Anomaly injection complete.")              # done message

output_filename = "AI_PES - Inferno AnomalyInjection/AI_PES - Inferno AnomalousDataset.xlsx" # output path
df_anomalous.to_excel(output_filename, index=False) # save file

print(f"Anomalous dataset saved to: {output_filename}") # final message