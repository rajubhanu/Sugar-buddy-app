import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("SugAR Buddy - Diabetes Risk Checker")

# User input
age = st.slider("Age", 18, 100)
bmi = st.number_input("BMI", 10.0, 50.0, step=0.1)
glucose = st.number_input("Glucose Level", 50, 200)
bp = st.number_input("Blood Pressure", 40, 130)
insulin = st.number_input("Insulin Level", 0.0, 500.0, step=0.1)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
pregnancies = st.slider("Pregnancies", 0, 15)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)

# Predict
if st.button("Predict Risk"):
    data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][prediction]

    st.subheader(f"🩺 Risk: {'High' if prediction == 1 else 'Low'} ({proba*100:.2f}%)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], data, matplotlib=True)
    st.pyplot(bbox_inches='tight')
