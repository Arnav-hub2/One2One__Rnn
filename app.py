import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# -------------------------
# Load Model
# -------------------------
model = load_model("model.h5")

scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="ECG Classification", layout="wide")

st.title("❤️ ECG Heartbeat Classification")
st.write("Enter 140 ECG values separated by commas.")
# -------------------------
# Dataset Information
# -------------------------

st.markdown("---")
st.subheader("📖 ECG5000 Class Information")

st.markdown("""
| **Class** | **Heartbeat Type** |
|-----------|--------------------|
| **0 / 1** | ❤️ Normal Heartbeat |
| **1 / 2** | 💛 Supraventricular Premature Beat (SVPB) |
| **2 / 3** | 🧡 Premature Ventricular Contraction (PVC) |
| **3 / 4** | 💜 Fusion of Ventricular and Normal Beat |
| **4 / 5** | ❤️‍🩹 Unknown / Unclassifiable Beat |
""")

st.info(
    "Note: During training, LabelEncoder converts the original dataset labels "
    "(1–5) into encoded labels (0–4). The predicted class shown by the model "
    "is mapped back to the original ECG5000 class before displaying."
)

st.markdown("---")

sample = st.text_area(
    "ECG Signal",
    height=200
)

if st.button("Predict"):

    try:

        values = [float(x) for x in sample.split(",")]

        if len(values) != 140:
            st.error("Please enter exactly 140 values.")
            st.stop()

        X = np.array(values).reshape(1,140)

        X = scaler.transform(X)

        X = X.reshape((1,140,1))

        prediction = model.predict(X)

        predicted = np.argmax(prediction)

        label = encoder.inverse_transform([predicted])[0]

        confidence = np.max(prediction)

        st.success(f"Predicted Class : {label}")

        st.info(f"Confidence : {confidence:.2%}")

    except Exception as e:

        st.error(e)