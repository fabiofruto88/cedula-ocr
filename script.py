import cv2
import pytesseract
import os
import logging
import numpy as np

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

def validar_imagen(ruta_imagen):
    """Valida que el archivo exista y sea una imagen."""
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encuentra la imagen en {ruta_imagen}")
    
    extension = os.path.splitext(ruta_imagen)[1].lower()
    if extension not in ['.jpg', '.jpeg', '.png', '.bmp']:
        raise ValueError(f"Formato de imagen no soportado: {extension}")

def mostrar_imagen_completa(titulo, imagen):
    """Muestra la imagen completa ajustando la ventana."""
    # Obtener dimensiones de la pantalla
    screen_res = 1366, 768  # Ajusta esto según tu resolución
    scale_width = screen_res[0] / imagen.shape[1]
    scale_height = screen_res[1] / imagen.shape[0]
    scale = min(scale_width, scale_height)
    
    # Calcular nueva dimensión
    window_width = int(imagen.shape[1] * scale)
    window_height = int(imagen.shape[0] * scale)
    
    # Configurar ventana
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(titulo, window_width, window_height)
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    cv2.destroyWindow(titulo)

def preprocesar_imagen(img):
    """Preprocesa la imagen preservando calidad y dimensiones."""
    # Mostrar imagen original completa
    mostrar_imagen_completa("1. Original", img)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mostrar_imagen_completa("2. Escala de grises", gray)
    
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrasted = clahe.apply(gray)
    mostrar_imagen_completa("3. Contraste mejorado", contrasted)
    
    # Reducir ruido manteniendo detalles
    denoised = cv2.fastNlMeansDenoising(contrasted, h=10)
    mostrar_imagen_completa("4. Reducción de ruido", denoised)
    
    # Umbral adaptativo suave
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        21, 11
    )
    mostrar_imagen_completa("5. Umbral adaptativo", thresh)
    
    # Guardar para verificación
    cv2.imwrite("preprocessed_image.png", thresh)
    return thresh  
    


def procesar_imagen(ruta_imagen):
    """Procesa la imagen y extrae el texto usando técnicas probadas."""
    validar_imagen(ruta_imagen)
    
    # Cargar imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    # Mostrar imagen original completa
    mostrar_imagen_completa("1. Imagen Original", img)
    
    # Preprocesamiento mínimo
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mostrar_imagen_completa("2. Escala de grises", gray)
    
    # Reducir ruido suavemente
    denoised = cv2.fastNlMeansDenoising(gray, h=10)  # h más bajo para preservar detalles
    mostrar_imagen_completa("3. Sin ruido", denoised)
    
    # Configuración OCR optimizada
    custom_config = r'--oem 3 --psm 3 -l spa'
    texto = pytesseract.image_to_string(denoised, config=custom_config)
    
    logger.debug("\n=== TEXTO EXTRAÍDO ===")
    logger.debug(texto)
    logger.debug("====================\n")
    
    return texto

def determinar_tipo_cedula(texto):
    """Determina el tipo de cédula basado en identificadores únicos."""
    texto = texto.upper()
    logger.debug("=== TEXTO EXTRAÍDO PARA ANÁLISIS ===")
    logger.debug(texto)
    logger.debug("=====================================")
    
    # Simplificar la detección a solo los identificadores únicos
    if "NUIP" in texto:
        return "nueva"
    elif "NÚMERO" in texto or "NUMERO" in texto:
        return "vieja"
    return "desconocido"

def extraer_datos_generales(texto, patrones):
    """Extrae datos usando patrones específicos."""
    datos = {}
    for campo, patron in patrones.items():
        for linea in texto.split('\n'):
            if patron.upper() in linea.upper():
                datos[campo] = linea.split(patron)[-1].strip()
                break
    return datos

def extraer_datos_vieja(texto):
    """Extrae datos de cédula vieja."""
    patrones = {
        "Número": "NÚMERO",
        "Apellidos": "APELLIDOS:",
        "Nombres": "NOMBRES:"
    }
    return extraer_datos_generales(texto, patrones)

def extraer_datos_nueva(texto):
    """Extrae datos de cédula nueva."""
    patrones = {
        "NUIP": "NUIP",
        "Apellidos": "APELLIDOS:",
        "Nombres": "NOMBRES:"
    }
    return extraer_datos_generales(texto, patrones)

def procesar_cedula(ruta_imagen):
    """Proceso principal de extracción de datos de la cédula."""
    try:
        texto = procesar_imagen(ruta_imagen)
        tipo_cedula = determinar_tipo_cedula(texto)
        logger.info("Tipo de cédula detectado: %s", tipo_cedula)
        
        if tipo_cedula == "vieja":
            datos = extraer_datos_vieja(texto)
        elif tipo_cedula == "nueva":
            datos = extraer_datos_nueva(texto)
        else:
            datos = {"Error": "No se pudo determinar el tipo de cédula."}
        
        return {"tipo_cedula": tipo_cedula, "datos": datos}
    
    except Exception as e:
        logger.error("Error procesando cédula: %s", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        # Ruta de la imagen de prueba
        ruta_imagen_prueba = "./cedula-vieja.jpg"
        
        # Procesar la cédula
        resultado = procesar_cedula(ruta_imagen_prueba)
        
        # Mostrar resultados
        print("\n=== Resultados del Procesamiento ===")
        print(f"Tipo de cédula: {resultado['tipo_cedula']}")
        print("\nDatos extraídos:")
        for campo, valor in resultado['datos'].items():
            print(f"{campo}: {valor}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
