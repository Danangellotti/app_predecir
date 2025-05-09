
import streamlit as st
import numpy as np
import joblib

# Cargar modelo entrenado
model = joblib.load("modelo_svm.pkl")  # El archivo .pkl debe estar en la misma carpeta

# Título
st.title("Predicción de Riesgo de Incendios")
st.write("Este modelo predice si un mes tiene riesgo de incendio bajo o moderado/alto según variables climáticas.")

# Entradas del usuario
rh = st.slider("Humedad relativa (RH)", 20, 100, 50)
wspd = st.slider("Velocidad del viento (km/h)", 0, 40, 15)
temp = st.slider("Temperatura (°C)", 0, 45, 25)

# Predicción
X_input = np.array([[rh, wspd, temp]])
pred = model.predict(X_input)[0]

# Mostrar resultado
if pred == 1:
    st.error("⚠️ Riesgo Predicho: MODERADO/ALTO")
else:
    st.success("✅ Riesgo Predicho: BAJO")
