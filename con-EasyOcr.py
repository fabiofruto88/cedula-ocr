import cv2
import easyocr
import os
import logging
import numpy as np
from PIL import Image

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar EasyOCR (solo una vez)
reader = easyocr.Reader(['es'], gpu=False)

def procesar_imagen(ruta_imagen):
    """Procesa la imagen y extrae el texto usando EasyOCR."""
    logger.debug(f"Procesando imagen: {ruta_imagen}")
    
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encuentra la imagen: {ruta_imagen}")
    
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    logger.debug("Imagen cargada correctamente")
    logger.debug(f"Dimensiones: {img.shape}")
    
    # Preprocesamiento básico
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Extraer texto con EasyOCR
    logger.debug("Iniciando extracción de texto...")
    resultados = reader.readtext(denoised)
    texto = '\n'.join([res[1] for res in resultados])
    
    logger.debug("Texto extraído:")
    logger.debug(texto)
    
    return texto

def extraer_datos_generales(texto, tipo_cedula):
    """Extrae datos según el tipo de cédula."""
    logger.debug(f"Extrayendo datos para cédula tipo: {tipo_cedula}")
    lineas = [linea.strip() for linea in texto.split('\n') if linea.strip()]
    datos = {}
    palabras_excluidas = ["NUIP", "NOMBRES", "APELLIDOS", "NÚMERO", "NUMERO"]
    
    if tipo_cedula == "vieja":
        # Primera lógica: Búsqueda por palabras clave
        for i, linea in enumerate(lineas):
            if "NÚMERO" in linea.upper() or "NUMERO" in linea.upper():
                if i + 1 < len(lineas):
                    datos["Número"] = lineas[i + 1].strip()
                    # Buscar apellidos en las siguientes líneas
                    for j in range(i + 2, len(lineas)):
                        siguiente_linea = lineas[j].strip()
                        if siguiente_linea and not any(palabra in siguiente_linea.upper() for palabra in palabras_excluidas):
                            datos["Apellidos"] = siguiente_linea
                            break
                    logger.debug(f"Número encontrado (por palabra clave): {datos.get('Número')}")
                break
        
        # Segunda lógica: Búsqueda por patrón si no se encontró
        if "Número" not in datos:
            for i, linea in enumerate(lineas):
                if any(c.isdigit() for c in linea) and '.' in linea:
                    datos["Número"] = linea.strip()
                    if i + 1 < len(lineas):
                        siguiente_linea = lineas[i + 1].strip()
                        if all(c.isupper() or c.isspace() for c in siguiente_linea):
                            datos["Apellidos"] = siguiente_linea
                    logger.debug(f"Número encontrado (por patrón): {datos.get('Número')}")
                    break

        # Buscar Nombres (mantener lógica original)
        for i, linea in enumerate(lineas):
            if "APELLIDOS" in linea.upper():
                if i + 1 < len(lineas):
                    siguiente_linea = lineas[i + 1].strip()
                    if not any(palabra in siguiente_linea.upper() for palabra in palabras_excluidas):
                        datos["Nombres"] = siguiente_linea
                        logger.debug(f"Nombres encontrados: {datos['Nombres']}")
                break
                
        # Si no se encontraron nombres por palabra clave, buscar después de apellidos
        if "Nombres" not in datos and "Apellidos" in datos:
            apellidos_index = next((i for i, linea in enumerate(lineas) if linea == datos["Apellidos"]), -1)
            if apellidos_index != -1 and apellidos_index + 1 < len(lineas):
                siguiente_linea = lineas[apellidos_index + 1].strip()
                if all(c.isupper() or c.isspace() for c in siguiente_linea):
                    datos["Nombres"] = siguiente_linea
                    logger.debug(f"Nombres encontrados (por posición): {datos['Nombres']}")

    elif tipo_cedula == "nueva":
        # Mantener lógica existente para cédula nueva
        for i, linea in enumerate(lineas):
            if "NUIP" in linea.upper():
                if i + 1 < len(lineas):
                    datos["NUIP"] = lineas[i + 1].strip()
                    if i + 2 < len(lineas):
                        siguiente_linea = lineas[i + 2].strip()
                        if siguiente_linea and not any(palabra in siguiente_linea.upper() for palabra in palabras_excluidas):
                            datos["Apellidos"] = siguiente_linea
                    logger.debug(f"NUIP encontrado: {datos['NUIP']}")
                break
        
        for i, linea in enumerate(lineas):
            if "NOMBRES" in linea.upper():
                if i + 1 < len(lineas):
                    siguiente_linea = lineas[i + 1].strip()
                    if not any(palabra in siguiente_linea.upper() for palabra in palabras_excluidas):
                        datos["Nombres"] = siguiente_linea
                        logger.debug(f"Nombres encontrados: {datos['Nombres']}")
                break
    
    return datos

def determinar_tipo_cedula(texto):
    """Determina el tipo de cédula."""
    texto = texto.upper()
    logger.debug("Determinando tipo de cédula...")
    
    if "NUIP" in texto:
        logger.debug("Detectada cédula nueva")
        return "nueva"
    # Agregamos "PERSONAL" como identificador adicional
    elif any(palabra in texto for palabra in ["NÚMERO", "NUMERO", "IDENTIFICACION PERSONAL", "IDENTIFICACIÓN PERSONAL"]):
        logger.debug("Detectada cédula vieja")
        return "vieja"
    
    logger.debug("Tipo de cédula desconocido")
    return "desconocido"

def procesar_cedula(ruta_imagen):
    """Proceso principal de extracción de datos de la cédula."""
    try:
        texto = procesar_imagen(ruta_imagen)
        tipo_cedula = determinar_tipo_cedula(texto)
        logger.info(f"Tipo de cédula detectado: {tipo_cedula}")
        
        datos = extraer_datos_generales(texto, tipo_cedula)
        logger.debug(f"Datos extraídos: {datos}")
        
        return {
            "tipo_cedula": tipo_cedula,
            "datos": datos,
            "texto_completo": texto
        }
    
    except Exception as e:
        logger.error(f"Error procesando cédula: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        ruta_imagen = "./cedula-vieja4.jpg"
        resultado = procesar_cedula(ruta_imagen)
        
        print("\n=== Resultados del Procesamiento ===")
        if "error" in resultado:
            print(f"Error: {resultado['error']}")
        else:
            print(f"Tipo de cédula: {resultado['tipo_cedula']}")
            print("\nDatos extraídos:")
            for campo, valor in resultado['datos'].items():
                print(f"{campo}: {valor}")
            print("\nTexto completo (debug):")
            print(resultado['texto_completo'])
            
    except Exception as e:
        print(f"Error: {str(e)}")