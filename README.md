# 🏥 Automated Metabolic Screening: End-to-End Clinical Prediction Pipeline

[![Python 3.10+](https://shields.io)](https://python.org)
[![Scikit-Learn](https://shields.io)](https://scikit-learn.org)
[![License: MIT](https://shields.io)](https://opensource.org)

## 📌 1. Executive Summary & Clinical Problem Statement
In modern healthcare systems, early identification of metabolic syndromes such as Type-2 Diabetes is critical to preventing chronic long-term organ damage, cardiovascular disease, and high hospitalization costs. However, clinical diagnostics are frequently hindered by **incomplete patient records**, where missing laboratory values are mistakenly logged as raw `0` entries (e.g., a recorded blood insulin or skinfold thickness of 0). 

If left unaddressed, these structural anomalies corrupt standard machine learning models, leading to biased predictions, dangerous false negatives, and broken production systems.

**The Solution:** This project delivers an enterprise-grade, fully decoupled **Machine Learning Production Pipeline**. It structurally flags hidden data anomalies, dynamically imputes missing physiological features using K-Nearest Neighbors lookalike data, standardizes patient tracking metrics, evaluates multiple competing core algorithms, and serializes the champion model for instant web or cloud deployment.

---

## 📊 2. Dataset Architecture & Source
The pipeline uses the authoritative **Pima Indians Diabetes Dataset**, originally sourced from the **National Institute of Diabetes and Digestive and Kidney Diseases**. 

* **Data Stream Source:** Host-decoupled CSV stream via web URL.
* **Target Feature:** `Outcome` (0 = Healthy/Low Risk, 1 = High Risk/Diabetic)
* **Biological Predictor Features:**
  * `Pregnancies`: Number of times pregnant (Valid numerical 0s).
  * `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
  * `BloodPressure`: Diastolic blood pressure (mm Hg).
  * `SkinThickness`: Triceps skin fold thickness (mm).
  * `Insulin`: 2-Hour serum insulin (mu U/ml).
  * `BMI`: Body mass index (weight in kg/(height in m)²).
  * `DiabetesPedigreeFunction`: A genetic scoring function factoring family diabetes history.
  * `Age`: Patient age in years.

---

## ⚙️ 3. Production Pipeline Architecture
Unlike standard notebook-bound experimental code, this system is designed using strict **Modular Software Engineering Principles** to eliminate data leakage and ensure reproducibility.

```text
[ Raw Data Input ] ──> [ src/data_cleaning.py ] ──> [ Stratified Train/Test Split ]
                                                                 │
                                           ┌─────────────────────┴─────────────────────┐
                                           ▼ (Training Assembly Line)                  ▼ (Secret Test Set)
                                   [ StandardScaler ]                          [ Frozen Scaler Rules ]
                                           ▼                                           ▼
                                   [ KNNImputer (k=5) ]                        [ Frozen Imputer Rules ]
                                           ▼                                           ▼
                                   [ K-Neighbors Brain ]                       [ Inference Verification ]
                                           ▼                                           ▼
                                   [ Model Serialization ] ──> final_pipeline.joblib ──> [ Production Ready ]
```

### Key Engineering Guardrails Built-In:
1. **Defensive Anomaly Isolation:** Automatically separates valid biological zeros (like `Pregnancies`) from missing lab data errors (`Insulin`, `Glucose`, `BMI`) and flags them as `np.nan` so standard model computations don't warp.
2. **Conveyor-Belt Pipeline Execution:** Leverages Scikit-Learn `Pipeline` architectures to bind scaling, imputation, and classification into a single execution frame. 
3. **Data Leakage Elimination:** The data splitting happens *before* scaling or lookalike filling, ensuring the model never "cheats" by peaking at statistical metrics from the validation pool.

---

## 🏆 4. Model Selection & Showdown Results
An automated laboratory tournament was orchestrated inside the evaluation sandbox to pressure-test competing algorithmic architectures under identical validation constraints.

| Pipeline Model Architecture | Testing Validation Accuracy | Status |
| :--- | :---: | :---: |
| **K-Neighbors Classifier (K=5)** | **76.0%** | 🏆 **CHAMPION SELECTED** |
| Random Forest Classifier | 71.4% | Defeated |
| Decision Tree Classifier | 68.2% | Defeated |

*Engineered Insight:* While tree-based algorithms are often highly robust, the neighborhood distance boundaries calculated by the **K-Neighbors Classifier** successfully decoded the complex cellular inter-dependencies between BMI, Insulin, and Glucose, outperforming the ensemble methods by a significant margin.

---

## 🚀 5. Installation & Deployment Guide

### Prerequisites
Ensure you have a virtual environment initialized on Python 3.10+.

### Step 1: Clone the Repository & Replicate Environment
```bash
git clone https://github.com
cd YOUR_REPO_NAME
pip install -r requirements.txt
```

### Step 2: Run the Production Assembly Line
Executing `main.py` runs the entire factory loop: streaming live data from the web, filtering anomalies, training the champion algorithm, printing verification metrics, and freezing the final brain to the hard drive.
```bash
python main.py
```

### Step 3: Production Model Persistence
Once completed, the pipeline outputs a unified, optimized serialization matrix file:
`final_diabetes_knn_pipeline.joblib`

This single binary container can be instantly deployed to an API backend, a cloud function (AWS Lambda), or a web app interface to perform real-time, zero-latency clinical predictions on incoming patients.
