from src.data_cleaning import load_and_flag_anomalies
from src.modeling import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

url = ("https://raw.githubusercontent.com/plotly/datasets/"
       "master/diabetes.csv")

# Step 1: Run the raw clean
X, y = load_and_flag_anomalies(url)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Use the chosen algorithm and assemble the pipeline
chosen_algorithm = KNeighborsClassifier(n_neighbors=5)
pipeline_engine = build_pipeline(chosen_algorithm)

# Step 4: Execute training and testing
pipeline_engine.fit(X_train, y_train)
test_accuracy = pipeline_engine.score(X_test, y_test)

print(f"Production Accuracy: {test_accuracy * 100:.2f}%")

# Step 5: Freeze trained pipeline permanently
joblib.dump(pipeline_engine, "final_diabetes_pipeline.joblib")
print("Pipeline saved successfully and ready for production deployment!")


# =====================================================================
# 🔮 LIVE PREDICTION ON A NEW PATIENT (Right here at the end!)
# =====================================================================
import numpy as np
import pandas as pd

new_patient = pd.DataFrame([{
    "Pregnancies": 1,
    "Glucose": 115,
    "BloodPressure": 72,
    "SkinThickness": 0,    
    "Insulin": 0,          
    "BMI": 30.1,
    "DiabetesPedigreeFunction": 0.25,
    "Age": 33
}])


prediction = pipeline_engine.predict(new_patient)
probabilities = pipeline_engine.predict_proba(new_patient)

print("\n" + "="*45)
if prediction == 1:
    print("🚨 DIAGNOSIS: HIGH RISK OF DIABETES (Class 1)")
else:
    print("🟢 DIAGNOSIS: HEALTHY / LOW RISK (Class 0)")
    
print(f"📊 Confidence Score: {(probabilities[0][prediction] * 100).item():.1f}%")
print("="*45)

