import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# Cargar modelo entrenado
model = joblib.load("modelo_svm.pkl")

# Configuración de página
st.set_page_config(page_title="Riesgo de Incendios", page_icon="🔥", layout="centered")

st.title("🔥 Predicción de Riesgo de Incendios")
st.markdown("Esta app predice si un mes presenta **riesgo bajo** o **riesgo moderado/alto** de incendio, basado en datos climáticos.")

st.markdown("---")
st.header("🌦️ Ingresá los valores climáticos")

rh = st.slider("Humedad Relativa (%)", 20, 100, 50)
wspd = st.slider("Velocidad del Viento (km/h)", 0, 40, 15)
temp = st.slider("Temperatura (°C)", 0, 45, 25)

st.markdown("---")
st.subheader("📊 Radar de variables ingresadas")

fig = go.Figure(data=go.Scatterpolar(
    r=[rh, wspd, temp],
    theta=['Humedad', 'Viento', 'Temperatura'],
    fill='toself'
))
fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
st.plotly_chart(fig)

st.markdown("---")

if st.button("🔍 Predecir Riesgo"):
    X_input = np.array([[rh, wspd, temp]])
    pred = model.predict(X_input)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_input)[0][1]
    else:
        prob = "No disponible"

    if pred == 1:
        st.error("⚠️ Riesgo Predicho: **MODERADO/ALTO**")
        st.markdown("Tener precaución. Podrían generarse condiciones favorables para incendios.")
    else:
        st.success("✅ Riesgo Predicho: **BAJO**")
        st.markdown("Condiciones estables, sin alerta de riesgo alto.")

    st.markdown(f"**Probabilidad de riesgo:** {prob if isinstance(prob, str) else f'{prob:.2%}'}")

    if 'historial' not in st.session_state:
        st.session_state.historial = []

    st.session_state.historial.append({
        'Humedad': rh,
        'Viento': wspd,
        'Temperatura': temp,
        'Predicción': 'MOD/ALTO' if pred == 1 else 'BAJO',
        'Probabilidad': prob if isinstance(prob, str) else round(prob, 4)
    })

st.markdown("---")

if 'historial' in st.session_state and st.session_state.historial:
    st.subheader("🕘 Historial de Predicciones")
    st.dataframe(st.session_state.historial)

st.markdown("---")
st.caption("Desarrollado por Dana Angellotti • Modelo SVM ajustado • Streamlit App")


