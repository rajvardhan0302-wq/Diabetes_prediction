import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered",
)

# ── Load scaler (and model if present) ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    model = None
    for name in ("model.pkl", "classifier.pkl", "diabetes_model.pkl"):
        if os.path.exists(name):
            model = joblib.load(name)
            break
    return scaler, model

scaler, model = load_artifacts()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Prediction")
st.markdown(
    "Enter the patient's details below and click **Predict** to check "
    "the likelihood of diabetes."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1, step=1,
        help="Number of times pregnant"
    )
    glucose = st.number_input(
        "Glucose (mg/dL)", min_value=0, max_value=300, value=120,
        help="Plasma glucose concentration (2-hour oral glucose tolerance test)"
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70,
        help="Diastolic blood pressure"
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)", min_value=0, max_value=100, value=20,
        help="Triceps skinfold thickness"
    )

with col2:
    insulin = st.number_input(
        "Insulin (µU/mL)", min_value=0, max_value=900, value=80,
        help="2-Hour serum insulin"
    )
    bmi = st.number_input(
        "BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f",
        help="Body mass index (weight in kg / height in m²)"
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
        value=0.5, format="%.3f",
        help="A function that scores the likelihood of diabetes based on family history"
    )
    age = st.number_input(
        "Age (years)", min_value=1, max_value=120, value=30, step=1
    )

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)

    if model is not None:
        prediction = model.predict(scaled_data)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_data)[0][1]

        st.subheader("Result")
        if prediction == 1:
            st.error("⚠️ **Diabetic** – The model predicts a positive diabetes diagnosis.")
        else:
            st.success("✅ **Non-Diabetic** – The model predicts no diabetes.")

        if proba is not None:
            st.metric("Probability of Diabetes", f"{proba * 100:.1f}%")
    else:
        # No model loaded — just show scaled values
        st.warning(
            "⚠️ No model file found. Place your trained model file "
            "(`model.pkl`, `classifier.pkl`, or `diabetes_model.pkl`) "
            "in the same directory as `app.py` and restart the app."
        )
        st.subheader("Scaled Input Features (preview)")
        feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        ]
        for name, val in zip(feature_names, scaled_data[0]):
            st.write(f"**{name}:** {val:.4f}")

st.divider()
st.caption(
    "⚕️ This tool is for educational/demonstration purposes only. "
    "Always consult a medical professional for health decisions."
)
