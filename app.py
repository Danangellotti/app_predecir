import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# Cargar modelo entrenado
model = joblib.load("modelo_svm.pkl")

# Configuración de página
st.set_page_config(page_title="Riesgo de Incendios", page_icon="🔥", layout="centered")

# Estilos personalizados
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔥 Predicción de Riesgo de Incendios")
st.markdown("Esta app predice si un mes presenta **riesgo bajo** o **riesgo moderado/alto** de incendio, basado en datos climáticos.")

tab1, tab2 = st.tabs(["📊 Predicción", "🧾 Historial"])

with tab1:
    st.markdown("### 🌦️ Ingresá los valores climáticos")

    rh = st.slider("Humedad Relativa (%)", 20, 100, 50)
    wspd = st.slider("Velocidad del Viento (km/h)", 0, 40, 15)
    temp = st.slider("Temperatura (°C)", 0, 45, 25)

    st.markdown("### 📊 Radar de variables ingresadas")
    fig = go.Figure(data=go.Scatterpolar(
        r=[rh, wspd, temp],
        theta=['Humedad', 'Viento', 'Temperatura'],
        fill='toself'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    st.plotly_chart(fig)

    if st.button("🔍 Predecir Riesgo"):
        X_input = np.array([[rh, wspd, temp]])
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else "No disponible"

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
            'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Humedad': rh,
            'Viento': wspd,
            'Temperatura': temp,
            'Predicción': 'MOD/ALTO' if pred == 1 else 'BAJO',
            'Probabilidad': prob if isinstance(prob, str) else round(prob, 4)
        })

        with st.expander("🧠 ¿Cómo se interpreta este resultado?"):
            st.write("El modelo utiliza un clasificador SVM entrenado con datos históricos de clima. Una predicción de 'mod/alto' sugiere que los valores actuales son similares a los registrados en meses con riesgo real de incendio.")

with tab2:
    if 'historial' in st.session_state and st.session_state.historial:
        st.subheader("🕘 Historial de Predicciones")
        st.dataframe(pd.DataFrame(st.session_state.historial))

        if st.button("💾 Exportar historial a CSV"):
            pd.DataFrame(st.session_state.historial).to_csv("historial_predicciones.csv", index=False)
            st.success("✅ Historial exportado como historial_predicciones.csv")

st.markdown("---")
st.caption("Desarrollado por Dana Angellotti • Modelo SVM ajustado • Streamlit App con interfaz extendidaa")



