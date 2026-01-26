import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration - MUST BE FIRST
st.set_page_config(page_title="Heart Attack Predictor", page_icon="❤️", layout="wide")

# Sidebar page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Heart Attack Risk Prediction", "Clinical Measurements Guide"]
)

import os

if page == "Heart Attack Risk Prediction":
    # Load the trained model and scaler
    @st.cache_resource
    def load_model():
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, "heart_attack_app", "heart_attack_model.pkl")
        scaler_path = os.path.join(base_path, "heart_attack_app", "scaler.pkl")

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)

        return model, scaler

    model, scaler = load_model()

    # Title and description
    st.title("❤️ Heart Attack Risk Prediction")
    st.markdown("""
    This application predicts the risk of heart attack based on clinical features.
    Please enter the patient's information below.
    """)

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox("Chest Pain Type", 
                          options=[(0, "Typical Angina"), (1, "Atypical Angina"), 
                                   (2, "Non-anginal Pain"), (3, "Asymptomatic")],
                          format_func=lambda x: x[1])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

    with col2:
        st.subheader("Clinical Measurements")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox("Resting ECG Results", 
                               options=[(0, "Normal"), (1, "ST-T Abnormality"), (2, "LV Hypertrophy")],
                               format_func=lambda x: x[1])
        thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Additional features
    col3, col4 = st.columns(2)
    with col3:
        slope = st.selectbox("Slope of Peak Exercise ST", 
                            options=[(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
                            format_func=lambda x: x[1])
    with col4:
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", 
                           options=[(1, "Normal"), (2, "Fixed Defect"), (3, "Reversible Defect")],
                           format_func=lambda x: x[1])

    # Predict button
    if st.button("🔍 Predict Heart Attack Risk", type="primary"):
        # Prepare input data
        # Create input dataframe matching training features
        input_df = pd.DataFrame([{
            'age': age,
            'trestbps': trestbps,
            'chol': chol,
            'thalch': thalch,
            'oldpeak': oldpeak,
            'ca': ca,
            'sex_Male': 1 if sex[1] == 1 else 0,
            'cp_atypical angina': 1 if cp[0] == 1 else 0,
            'cp_non-anginal': 1 if cp[0] == 2 else 0,
            'cp_typical angina': 1 if cp[0] == 0 else 0,
            'fbs_True': 1 if fbs[1] == 1 else 0,
            'restecg_normal': 1 if restecg[0] == 0 else 0,
            'restecg_st-t abnormality': 1 if restecg[0] == 1 else 0,
            'exang_True': 1 if exang[1] == 1 else 0,
            'slope_flat': 1 if slope[0] == 1 else 0,
            'slope_upsloping': 1 if slope[0] == 0 else 0,
            'thal_normal': 1 if thal[0] == 1 else 0,
            'thal_reversable defect': 1 if thal[0] == 3 else 0
        }])

        TRAINING_COLUMNS = [
            'age','trestbps','chol','thalch','oldpeak','ca',
            'sex_Male','cp_atypical angina','cp_non-anginal','cp_typical angina',
            'fbs_True','restecg_normal','restecg_st-t abnormality','exang_True',
            'slope_flat','slope_upsloping','thal_normal','thal_reversable defect'
        ]

        input_df = input_df[TRAINING_COLUMNS]

        # Scale only numerical columns (same as training)
        num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error("⚠️ **HIGH RISK** - Patient has a high risk of heart attack")
            risk_percentage = prediction_proba[0][1] * 100
        else:
            st.success("✅ **LOW RISK** - Patient has a low risk of heart attack")
            risk_percentage = prediction_proba[0][0] * 100
        
        # Display probability
        st.metric("Confidence Level", f"{risk_percentage:.2f}%")
        
        # Display feature importance or additional info
        st.info("💡 **Note**: This is a predictive model and should not replace professional medical advice. Please consult a healthcare provider for accurate diagnosis.")

    # Add sidebar with information
    st.sidebar.header("About")
    st.sidebar.info("""
    This Heart Attack Risk Prediction model uses Machine Learning 
    (Logistic Regression) trained on clinical data to assess risk.

    **Features Used:**
    - Age, Sex, Chest Pain Type
    - Blood Pressure & Cholesterol
    - ECG Results
    - Exercise Metrics
    - And more...

    **Model Performance:**
    - Accuracy: 0.54
    - Precision: 0.48
    - Recall: 0.54
    """)

    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Enter patient information
    2. Fill in all clinical measurements
    3. Click 'Predict' button
    4. View risk assessment
    """)

elif page == "Clinical Measurements Guide":
    st.title("📘 Clinical Measurements Guide")
    st.markdown("""
This page explains the meaning of each clinical measurement used in the
Heart Attack Risk Prediction model. It helps new users understand
what values to enter in the patient information form.
""")

    st.markdown("### 🧍 Patient Information")

    st.markdown("""
**Age**  
Patient's age in years.

**Sex**  
Biological sex of the patient.

**Chest Pain Type (CP)**  
- Typical Angina: Chest pain related to reduced blood flow to the heart  
- Atypical Angina: Chest pain not following classic patterns  
- Non-anginal Pain: Chest pain unrelated to the heart  
- Asymptomatic: No chest pain
""")

    st.markdown("### 🩺 Clinical Measurements")

    st.markdown("""
**Resting Blood Pressure (trestbps)**  
Blood pressure measured in mm Hg while at rest.

**Cholesterol (chol)**  
Serum cholesterol level in mg/dl.

**Fasting Blood Sugar (fbs)**  
Whether fasting blood sugar is greater than 120 mg/dl.

**Resting ECG (restecg)**  
Electrocardiogram results at rest:
- Normal  
- ST-T wave abnormality  
- Left ventricular hypertrophy

**Maximum Heart Rate Achieved (thalch)**  
Highest heart rate achieved during exercise.

**Exercise Induced Angina (exang)**  
Whether chest pain occurs during exercise.

**ST Depression (oldpeak)**  
Depression of the ST segment induced by exercise.

**Slope of Peak Exercise ST (slope)**  
- Upsloping  
- Flat  
- Downsloping

**Number of Major Vessels (ca)**  
Number of major blood vessels colored by fluoroscopy (0–3).

**Thalassemia (thal)**  
Blood disorder classification:
- Normal  
- Fixed defect  
- Reversible defect
""")

    st.info("ℹ️ This information is for educational purposes only and should not replace medical advice.")
