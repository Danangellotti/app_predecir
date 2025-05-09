import streamlit as st
import numpy as np
import joblib

# Cargar modelo entrenado
model = joblib.load("modelo_svm.pkl")

# Configuración de página
st.set_page_config(page_title="Riesgo de Incendios", page_icon="🔥", layout="centered")

# Título con emoji
st.title("🔥 Predicción de Riesgo de Incendios")
st.markdown("Esta app predice si un mes presenta **riesgo bajo** o **riesgo moderado/alto** de incendio, basado en datos climáticos.")

# Línea separadora
st.markdown("---")

# 🎛️ Sliders de entrada
st.header("🌦️ Ingresá los valores climáticos")

rh = st.slider("Humedad Relativa (%)", min_value=20, max_value=100, value=50, step=1)
wspd = st.slider("Velocidad del Viento (km/h)", min_value=0, max_value=40, value=15, step=1)
temp = st.slider("Temperatura (°C)", min_value=0, max_value=45, value=25, step=1)

# Línea separadora
st.markdown("---")

# Botón de predicción
if st.button("🔍 Predecir Riesgo"):
    X_input = np.array([[rh, wspd, temp]])
    pred = model.predict(X_input)[0]

    if pred == 1:
        st.error("⚠️ Riesgo Predicho: **MODERADO/ALTO**")
        st.markdown("Tener precaución. Podrían generarse condiciones favorables para incendios.")
    else:
        st.success("✅ Riesgo Predicho: **BAJO**")
        st.markdown("Condiciones estables, sin alerta de riesgo alto.")

# Footer
st.markdown("---")
st.caption("Desarrollado por Dana Angellotti • Modelo SVM ajustado • Streamlit App")

