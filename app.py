import streamlit as st
import joblib
import numpy as np

# Title
st.title("Breast Cancer Prediction using SVM")

st.write("Click the button below to test prediction on sample data")

# Load model and scaler
model = joblib.load("svm_model/svm_breast_cancer.joblib")
scaler = joblib.load("svm_model/scaler_breast_cancer.pkl")

# Sample data
sample_data = np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
                         1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
                         25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1183]])

# Button
if st.button("Predict"):
    scaled = scaler.transform(sample_data)
    prediction = model.predict(scaled)[0]

    if prediction == 0:
        st.error("Result: Malignant (Cancerous)")
    else:
        st.success("Result: Benign (Non-Cancerous)")