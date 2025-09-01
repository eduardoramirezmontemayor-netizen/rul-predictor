import streamlit as st
import pickle
import numpy as np

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

# Botón de predicción
if st.button("🔮 Predecir RUL"):
    entrada = np.array(valores).reshape(1, -1)
    prediccion = model.predict(entrada)[0]
    st.success(f"✅ Predicción de RUL: {prediccion:.2f} ciclos restantes")

    # Mensaje de alerta si el RUL es bajo
    if prediccion < 30:
        st.warning("⚠️ Atención: el motor está cerca del final de su vida útil.")
