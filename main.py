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

# Verificar si el directorio existe antes de listar archivos
if not os.path.exists(directorio_pruebas):
    st.error(f"El directorio {directorio_pruebas} no existe. Verifica la ruta.")
else:
    # Lista todos los archivos en el directorio de pruebas
    archivos = os.listdir(directorio_pruebas)
    
    # Lista para almacenar las rutas de las imágenes
    rutas_imagenes = []
    
    # Obtener las rutas de las imágenes y guardarlas en la lista
    for archivo in archivos:
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            imagen_path = os.path.join(directorio_pruebas, archivo)
            rutas_imagenes.append(imagen_path)
    resultados = []
    df_resultados = pd.DataFrame(resultados)
    # Procesar y predecir para cada imagen en el directorio de pruebas
    if st.button('Presionar aquí para predecir'):
        if rutas_imagenes:
            barra_progreso = st.progress(0.0)
            progreso_texto = st.empty()
            for idx, ruta_imagen in enumerate(rutas_imagenes, start=1):
                ruta = cv2.imread(ruta_imagen)
                nombre_imagen = os.path.basename(ruta_imagen)
                imagen_procesada = cargar_y_preprocesar_imagen(ruta)
                prediccion = modelo.predict(imagen_procesada)
                resultado = 1 if prediccion[0][0] >= 0.5 else 0
                resultados.append({'ID': nombre_imagen, 'score': resultado})
                avance = idx / len(rutas_imagenes)
                barra_progreso.progress(avance)
                
                # Mostrar el porcentaje avanzado con decimales
                progreso_texto.text(f'Progreso: {(avance*100):.2f}%')
                
            df_resultados = pd.DataFrame(resultados)
            st.write("Resultados de las predicciones:")
            st.dataframe(df_resultados)
            nombre_archivo = 'resultados_predicciones.csv'
            csv = df_resultados.to_csv(index=False)
            # Generar el botón de descarga
            st.download_button(label='Descargar CSV', data=csv, file_name=nombre_archivo, mime='text/csv')

            
        else:
            st.write("No se encontraron imágenes en el directorio especificado.")
