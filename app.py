import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_lottie import st_lottie
import requests

# Set page configuration
st.set_page_config(page_title="Health Predictor", page_icon="🏥", layout="centered")

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load assets
lottie_health = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp9vbg.json")

# Load the trained model
@st.cache_resource
def load_model():
    with open("model (1).pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Custom CSS for animation and styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
with st.container():
    st.title("Diabetes Risk Assessment 🩺")
    st.write("Enter the patient details below to get a prediction.")
    if lottie_health:
        st_lottie(lottie_health, height=200, key="coding")

st.write("---")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submit_button = st.form_submit_button(label="Analyze Results")

# Prediction Logic
if submit_button:
    # Prepare the features for prediction
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    st.write("---")
    
    if prediction[0] == 1:
        st.error(f"### High Risk Detected")
        st.write(f"The model indicates a positive result with a probability of {prediction_proba[0][1]:.2%}")
    else:
        st.success(f"### Low Risk Detected")
        st.write(f"The model indicates a negative result with a probability of {prediction_proba[0][0]:.2%}")
        
    st.info("Note: This is a machine learning prediction and should not replace professional medical advice.")
