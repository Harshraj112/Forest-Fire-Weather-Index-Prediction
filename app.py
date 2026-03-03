import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── Page Config ───
st.set_page_config(
    page_title="FWI Prediction System",
    page_icon="🔥",
    layout="centered",
)

# ─── Load Models ───
@st.cache_resource
def load_models():
    ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    return ridge_model, scaler

ridge_model, standard_scaler = load_models()

# ─── Mappings ───
CLASSES_MAP = {"Fire": 1, "Not Fire": 0}
REGION_MAP = {"Bejaia": 0, "Sidi Bel-Abbes": 1}

# ─── UI ───
st.title("🔥 Forest Fire Weather Index (FWI) Prediction")
st.markdown(
    "Predict the **Fire Weather Index** based on meteorological & fire-related features "
    "using a **Ridge Regression** model trained on the Algerian Forest Fires dataset."
)
st.divider()

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Temperature (°C)", value=25.0, format="%.1f")
        rh = st.number_input("Relative Humidity (%)", value=60.0, format="%.1f")
        ws = st.number_input("Wind Speed (km/h)", value=15.0, format="%.1f")
        rain = st.number_input("Rainfall (mm)", value=0.0, min_value=0.0, format="%.2f")

    with col2:
        ffmc = st.number_input("FFMC Index", value=85.0, format="%.1f", help="Fine Fuel Moisture Code")
        dmc = st.number_input("DMC Index", value=25.0, format="%.1f", help="Duff Moisture Code")
        isi = st.number_input("ISI Index", value=5.0, format="%.1f", help="Initial Spread Index")
        classes = st.selectbox("Fire Class", options=list(CLASSES_MAP.keys()))
        region = st.selectbox("Region", options=list(REGION_MAP.keys()))

    submitted = st.form_submit_button("🔥 Predict FWI", use_container_width=True)

if submitted:
    classes_encoded = CLASSES_MAP[classes]
    region_encoded = REGION_MAP[region]

    features = np.array(
        [[temperature, rh, ws, rain, ffmc, dmc, isi, classes_encoded, region_encoded]]
    )
    scaled_features = standard_scaler.transform(features)
    prediction = ridge_model.predict(scaled_features)

    st.divider()
    st.metric(label="Predicted FWI", value=f"{prediction[0]:.2f}")
    
    if prediction[0] > 15:
        st.error("⚠️ High fire danger! Extreme caution recommended.")
    elif prediction[0] > 5:
        st.warning("🟡 Moderate fire danger. Stay alert.")
    else:
        st.success("🟢 Low fire danger.")

# ─── Sidebar ───
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **FWI Prediction System** uses a Ridge Regression model 
        trained on the **Algerian Forest Fires** dataset to predict 
        the Fire Weather Index.

        **Input Features:**
        - Temperature, Humidity, Wind Speed, Rainfall
        - FFMC, DMC, ISI (fire indices)
        - Fire Class & Region

        **Model:** Ridge Regression with StandardScaler

        [GitHub Repository](https://github.com/Harshraj112/Linear_regression_project)
        """
    )
