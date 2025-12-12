import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Layout
st.set_page_config(page_title="Diabetes Predictor", layout="centered")


# --- Load pipeline objects
@st.cache_resource
def load_model_objects():
    model = joblib.load("best_model.pkl")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, imputer, scaler

model, imputer, scaler = load_model_objects()

# --- Feature ranges
feature_ranges = {
    'Pregnancies': (0, 17),
    'Glucose': (0, 199),
    'BloodPressure': (0, 122),
    'SkinThickness': (0, 99),
    'Insulin': (0, 846),
    'BMI': (0, 67.1),
    'DiabetesPedigreeFunction': (0, 2.5),
    'Age': (21, 81)
}



st.title("Diabetes Prediction App")
st.markdown("""
This app uses a machine learning model trained on the **Pima Indians Diabetes dataset**  
to predict whether a person is diabetic based on medical parameters.
""")

# --- Sidebar
st.sidebar.header(" Feature Info")
for feature, (min_val, max_val) in feature_ranges.items():
    st.sidebar.markdown(
        f"**{feature}**\nRange: [{min_val} - {max_val}]"
    )



# --- User Inputs
st.header("Enter Patient Data")

user_input = {}
for feature, (min_val, max_val) in feature_ranges.items():
    tooltip = None
    if feature == "Glucose":
        tooltip = "Plasma glucose concentration (mg/dL)."
    elif feature == "BloodPressure":
        tooltip = "Diastolic blood pressure (mm Hg)."
    elif feature == "SkinThickness":
        tooltip = "Triceps skinfold thickness (mm)."
    elif feature == "Insulin":
        tooltip = "Serum insulin (mu U/ml)."
    elif feature == "BMI":
        tooltip = "Body mass index (kg/m²)."
    elif feature == "DiabetesPedigreeFunction":
        tooltip = "Likelihood of diabetes based on family history."
    elif feature == "Age":
        tooltip = "Age in years."
    elif feature == "Pregnancies":
        tooltip = "Number of times pregnant."

    value_str = st.text_input(
        label=f"{feature} [{min_val}-{max_val}]",
        help=tooltip,
        placeholder=f"Enter value between {min_val} and {max_val}"
    )

    # Handle empty or invalid input
    if value_str == "":
        value = np.nan
    else:
        try:
            value = float(value_str)
            # Check bounds
            if value < min_val or value > max_val:
                st.warning(f"⚠️ {feature} should be between {min_val} and {max_val}. Setting to NaN.")
                value = np.nan
        except ValueError:
            st.warning(f"⚠️ Invalid entry for {feature}. Setting to NaN.")
            value = np.nan

    # Interpret zeros as missing for medical variables
    if feature in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if value == 0:
            value = np.nan

    user_input[feature] = value


# --- Predict Button
if st.button("Predict Diabetes Risk"):
    try:
        # Create DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Impute missing
        input_df_imputed = pd.DataFrame(
            imputer.transform(input_df),
            columns=input_df.columns
        )
        
        # Scale
        input_scaled = scaler.transform(input_df_imputed)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)[0][1]
        else:
            prob = None
        
        # Display result
        if prediction == 1:
            st.error(f"**The person is likely Diabetic!**")
        else:
            st.success(f" **The person is likely Not Diabetic.**")
        
        if prob is not None:
            st.markdown(f"**Probability of being diabetic:** `{prob:.2%}`")

        # Export result
        download_df = pd.concat(
            [input_df_imputed, pd.DataFrame({"Prediction": [prediction], "Probability": [prob]})],
            axis=1
        )
        csv = download_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Prediction as CSV",
            data=csv,
            file_name="prediction_result.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f" An error occurred: {e}")

# --- Option to Reset
if st.button("Reset Inputs"):
    st.experimental_rerun()

# --- Optional Charts
st.header(" Data Insights (Optional)")

if st.checkbox("Show Example Prediction"):
    st.info("Example Prediction:")
    example = {k: (minv+maxv)/2 for k,(minv,maxv) in feature_ranges.items()}
    st.write(example)
