import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write("This app predicts diabetes using a **proper leakage-free ML pipeline** trained on the Pima Indian Diabetes dataset.")

st.markdown("---")

# ---------------------------------------------------------
# Load trained pipeline
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline():
    # Get path relative to this file (works locally + Streamlit Cloud)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "final_pipeline.pkl")

    pipeline = joblib.load(model_path)
    return pipeline

pipeline = load_pipeline()

# ---------------------------------------------------------
# Feature ranges
# ---------------------------------------------------------
feature_ranges = {
    'Pregnancies': (0, 17),
    'Glucose': (0, 199),
    'BloodPressure': (0, 122),
    'SkinThickness': (0, 99),
    'Insulin': (0, 846),
    'BMI': (0.0, 67.1),
    'DiabetesPedigreeFunction': (0.0, 2.5),
    'Age': (21, 81)
}

# Tooltip descriptions
tooltips = {
    "Pregnancies": "Number of times pregnant.",
    "Glucose": "Plasma glucose concentration after 2 hours.",
    "BloodPressure": "Diastolic blood pressure (mm Hg).",
    "SkinThickness": "Triceps skinfold thickness (mm).",
    "Insulin": "Serum insulin (mu U/ml).",
    "BMI": "Body mass index (kg/m²).",
    "DiabetesPedigreeFunction": "Likelihood of diabetes based on family history.",
    "Age": "Age in years."
}

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.header("Enter Patient Information")

user_input = {}

cols = st.columns(2)
i = 0

for feature, (min_val, max_val) in feature_ranges.items():
    col = cols[i % 2]
    i += 1

    value = col.number_input(
        label=f"{feature} ({min_val}-{max_val})",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(min_val),
        help=tooltips.get(feature, "")
    )

    # Convert zero to NaN for medically invalid zero columns
    if feature in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if value == 0:
            value = np.nan

    user_input[feature] = value

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
st.markdown("---")
if st.button("🔍 Predict Diabetes"):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([user_input])

        # Run through unified pipeline
        prediction = pipeline.predict(input_df)[0]

        # Probability
        prob = None
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(input_df)[0][1]

        # --- Display Result (Simple & Clean for All Users)
        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown("""
            <div style="background-color:#ffe6e6; padding:15px; border-radius:10px; border-left:8px solid #ff4d4d;">
                <h3 style="color:#b30000; margin:0;">⚠️ The person is likely <strong>Diabetic</strong>.</h3>
                <p style="color:#b30000; margin:5px 0 0 0;">Please consult a medical professional for confirmation.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background-color:#e6ffe6; padding:15px; border-radius:10px; border-left:8px solid #2ecc71;">
                <h3 style="color:#1e7e34; margin:0;">🟢 The person is likely <strong>Not Diabetic</strong>.</h3>
                <p style="color:#1e7e34; margin:5px 0 0 0;">Maintain a healthy lifestyle to reduce future risk.</p>
            </div>
            """, unsafe_allow_html=True)

        # Probability display (clean)
        if prob is not None:
            st.markdown(
                f"<h4 style='margin-top:20px;'>Probability of being diabetic: "
                f"<span style='color:#16a085;'>{prob*100:.2f}%</span></h4>",
                unsafe_allow_html=True
            )


        # Allow CSV download
        download_df = input_df.copy()
        download_df["Prediction"] = prediction
        download_df["Probability"] = prob

        st.download_button(
            label="📥 Download Result as CSV",
            data=download_df.to_csv(index=False).encode(),
            file_name="diabetes_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Developed by Rakhi Nehete • Leakage-Free ML Pipeline • Streamlit App")
