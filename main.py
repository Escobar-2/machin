import os
import streamlit as st
import requests
from tensorflow.keras.models import load_model

# Carpeta de las imágenes de prueba
directorio_pruebas = 'test'

# URL pública de descarga desde Google Drive
url = 'https://drive.google.com/uc?id=1GHc4_s-WtwA_04Pa2luZLP3sIy7akBIq'

# Nombre del archivo local donde se guardará el modelo
archivo_modelo = 'modelo.keras'

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

    
    # Función para cargar y preprocesar una imagen
    def cargar_y_preprocesar_imagen(ruta_imagen):
        imagen = cv2.imread(ruta_imagen)
        imagen = cv2.resize(imagen, (512, 512))
        imagen = imagen.astype('float32') / 255.0
        imagen = np.expand_dims(imagen, axis=-1)
        imagen = np.expand_dims(imagen, axis=0)
    
        return imagen
    

    # Función para descargar el archivo desde Google Drive
    def descargar_desde_google_drive(url, archivo_destino):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(archivo_destino, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            return False
    
    # Descargar el archivo del modelo desde Google Drive si no está localmente
    if not os.path.exists(archivo_modelo):
        if descargar_desde_google_drive(url, archivo_modelo):
            st.write(f"Archivo descargado correctamente como {archivo_modelo}")
            # Cargar el modelo desde el archivo descargado
            modelo = load_model(archivo_modelo)
            
            # Ejemplo de uso del modelo en Streamlit
            st.write("Modelo cargado correctamente. Puedes comenzar a hacer predicciones.")
        else:
            st.error("Error al descargar el archivo desde Google Drive. Verifica la URL.")
    
    
    # Imprimir las rutas de las imágenes
    if rutas_imagenes:
        st.write("Imagenes cargadas correctamente")
    else:
        st.write("No se encontraron imágenes en el directorio especificado.")
