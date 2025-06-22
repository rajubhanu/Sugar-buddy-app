import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.set_page_config(page_title="SugAR Buddy - డయాబెటిస్ చెక్", layout="centered")
st.title("SugAR Buddy - డయాబెటిస్ రిస్క్ చెకర్")

# User input in Telugu
age = st.slider("వయస్సు (Age)", 18, 100)
bmi = st.number_input("బిఎమ్‌ఐ (BMI)", 10.0, 50.0, step=0.1)
glucose = st.number_input("గ్లూకోజ్ లెవల్", 50, 200)
bp = st.number_input("బ్లడ్ ప్రెషర్", 40, 130)
insulin = st.number_input("ఇన్సులిన్ స్థాయి", 0.0, 500.0, step=0.1)
skin_thickness = st.number_input("చర్మం మందం", 0, 100)
pregnancies = st.slider("గర్భధారణల సంఖ్య", 0, 15)
dpf = st.number_input("కుటుంబ చరిత్ర (DPF)", 0.0, 2.5)

# Predict
if st.button("రిస్క్ తెలుసుకోండి"):
    data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(data)[0]
    proba = model.predict_proba(data)[0][prediction]

    result = "అధిక రిస్క్" if prediction == 1 else "తక్కువ రిస్క్"
    st.subheader(f"🩺 ఫలితం: {result} ({proba*100:.2f}%)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1], data, matplotlib=True)
    st.pyplot(bbox_inches='tight')
