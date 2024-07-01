import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

# Cargar el modelo
modelo = load_model('modelo_eficiente1.keras')

# Carpeta de las imágenes de prueba
directorio_pruebas = 'test'

# Función para cargar y preprocesar una imagen
def cargar_y_preprocesar_imagen(imagen):
    imagen = cv2.resize(imagen, (512, 512))
    imagen = imagen.astype('float32') / 255.0
    imagen = np.expand_dims(imagen, axis=0)  # No es necesario expandir el eje de canal si la imagen es a color
    return imagen

# Título y descripción en la interfaz de usuario
st.title('Proyecto final Machine Learning')
st.write('Detección de retinopatías diabéticas.')

# Cargar imágenes subidas por el usuario
uploaded_files = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        imagen_preparada = cargar_y_preprocesar_imagen(image)
        prediccion = modelo.predict(imagen_preparada)
        resultado = 1 if prediccion[0][0] >= 0.5 else 0
        original = Image.open(uploaded_file)
        st.image(original, caption=f'Predicción: {resultado}', use_column_width=True)

