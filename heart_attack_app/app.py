import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('heart_attack_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model()

# Set page configuration
st.set_page_config(page_title="Heart Attack Predictor", page_icon="❤️", layout="wide")

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
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Additional features (adjust based on your actual dataset)
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
    input_data = np.array([[age, sex[1], cp[0], trestbps, chol, fbs[1], 
                           restecg[0], thalach, exang[1], oldpeak, 
                           slope[0], ca, thal[0]]])
    
    # Scale the input if you used a scaler during training
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    
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