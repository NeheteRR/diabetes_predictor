# 🩺 Diabetes Prediction Using Machine Learning
ML Pipeline • Random Forest Model • Streamlit Web App
## 📌 Project Overview
This project predicts whether a person is diabetic based on medical parameters from the Pima Indians Diabetes Dataset.

It includes:

- A research-based methodology (baseline ML pipeline)
- A fully improved, leakage-free ML pipeline (best practice approach)
- A unified ML pipeline model saved as .pkl
- A Streamlit web app for real-time predictions

The goal is to build a reliable and production-ready classification system that avoids common mistakes like data leakage, incorrect SMOTE usage, or training–test contamination.
---
## 🚀 Features
🔬 Machine Learning Pipelines
- Data cleaning + missing value handling
- Zero → NaN transformation for medically invalid values
- KNN Imputation
- Standard Scaling
- SMOTE applied only on training data
- Multiple ML models tested (LR, DT, RF, NB, SVM, LGBM, XGBoost)
- Hyperparameter tuning using RandomizedSearchCV
- Final Random Forest model chosen based on G-Mean and AUC
- Streamlit Webapplication



📂 Project Structure
```bash
Diabetes_Prediction/
│
├── app/
│   └── diabetes_app.py           # Streamlit user interface
│
├── models/
│   └── final_pipeline.pkl        # Unified pipeline (imputer + scaler + model)
│
├── notebooks/
│   ├── research_methodology.ipynb
│   └── improved_leakage_free_method.ipynb
│
├── data/
│   └── diabetes.csv
│
├── results/
│   └── tuned_models_test_results.csv
│
├── requirements.txt
└── README.md
```

## 🛠 Installation & Setup
1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/diabetes-prediction.git
cd diabetes-prediction
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run Streamlit App
```bash
streamlit run app/diabetes_app.py
```