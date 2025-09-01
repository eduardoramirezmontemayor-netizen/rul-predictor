import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el modelo ultraligero
with open("modelo_RUL_ultraligero.pkl", "rb") as f:
    model = pickle.load(f)

# Configuración de la página
st.set_page_config(page_title="Predicción de RUL", page_icon="🛠️")
st.title("🛠️ Predicción de Vida Útil Restante (RUL) del Motor")
st.markdown("""
Esta aplicación estima cuántos ciclos le quedan a un motor de avión antes de fallar, 
basándose en los valores de sus sensores. Ideal para mantenimiento predictivo.
""")

# Lista de sensores usados en el modelo
sensores = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

# Formulario de entrada
st.subheader("🔍 Introduce los valores de los sensores")
valores = []
cols = st.columns(3)
for i, sensor in enumerate(sensores):
    with cols[i % 3]:
        valor = st.number_input(f"{sensor}", value=0.0, format="%.2f")
        valores.append(valor)

# Inicializar historial
if "historial" not in st.session_state:
    st.session_state.historial = []

# Botón de predicción
if st.button("🔮 Predecir RUL"):
    entrada = np.array(valores).reshape(1, -1)
    prediccion = model.predict(entrada)[0]
    st.success(f"✅ Predicción de RUL: {prediccion:.2f} ciclos restantes")

    if prediccion < 30:
        st.warning("⚠️ Atención: el motor está cerca del final de su vida útil.")

    # Guardar en historial
    st.session_state.historial.append({
        "RUL": round(prediccion, 2),
        **{sensor: val for sensor, val in zip(sensores, valores)}
    })

# Mostrar historial de predicciones
if st.session_state.historial:
    st.subheader("📋 Historial de predicciones")
    st.dataframe(pd.DataFrame(st.session_state.historial))

# Visualizar importancia de sensores
st.subheader("📊 Importancia de cada sensor en la predicción")

try:
    importancias = model.feature_importances_
    df_importancia = pd.DataFrame({
        'Sensor': sensores,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=True)

    fig, ax = plt.subplots()
    ax.barh(df_importancia['Sensor'], df_importancia['Importancia'], color='teal')
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de sensores")
    st.pyplot(fig)

except AttributeError:
    st.info("ℹ️ Este modelo no proporciona información de importancia de características.")
