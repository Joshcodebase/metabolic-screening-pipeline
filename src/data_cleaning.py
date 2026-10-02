import pandas as pd
import numpy as np

def load_and_flag_anomalies(file_path):
    """Loads dataset and converts impossible zeros to NaNs."""
    df = pd.read_csv(file_path)
    
    # The columns where 0 is an anomaly
    columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[columns_to_fix] = df[columns_to_fix].replace(0, np.nan)
    
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    return X, y


