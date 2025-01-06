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
def es_dato_valido(texto, tipo="general"):
    """Valida si el texto extraído cumple con criterios mínimos."""
    if not texto:
        return False
        
    if tipo == "numero":
        # Validar que tenga números y puntos
        return any(c.isdigit() for c in texto) and '.' in texto
    
    elif tipo == "apellidos":
        # Validación estricta para apellidos
        return (len(texto) > 4 and 
                all(c.isupper() or c.isspace() for c in texto) and
                not any(c.isdigit() for c in texto))
    
    elif tipo == "nombres":
        # Validación más flexible para nombres
        return (len(texto) >= 2 and 
                not any(c.isdigit() for c in texto) and
                ' ' in texto)  # Al menos debe tener un espacio (nombre y apellido)
    
    return True

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
        # Buscar número
        for i, linea in enumerate(lineas):
            if "NÚMERO" in linea.upper() or "NUMERO" in linea.upper():
                if i + 1 < len(lineas):
                    numero = lineas[i + 1].strip()
                    if es_dato_valido(numero, "numero"):
                        datos["Número"] = numero
                        logger.debug(f"Número encontrado: {datos['Número']}")
                break
        
        # Segunda búsqueda de número si no se encontró
        if "Número" not in datos:
            for i, linea in enumerate(lineas):
                if es_dato_valido(linea, "numero"):
                    datos["Número"] = linea.strip()
                    logger.debug(f"Número encontrado por patrón: {datos['Número']}")
                    break

        # Buscar apellidos por palabra clave
        for i, linea in enumerate(lineas):
            if "APELLIDOS" in linea.upper():
                for j in range(i + 1, len(lineas)):
                    siguiente_linea = lineas[j].strip()
                    if all(c.isupper() or c.isspace() for c in siguiente_linea) and len(siguiente_linea) > 4:
                        datos["Apellidos"] = siguiente_linea
                        logger.debug(f"Apellidos encontrados por palabra clave: {siguiente_linea}")
                        break
                break

        # Si no se encontraron apellidos, buscar después del número
        if "Apellidos" not in datos and "Número" in datos:
            for i, linea in enumerate(lineas):
                if linea.strip() == datos["Número"]:
                    for j in range(i + 1, len(lineas)):
                        siguiente_linea = lineas[j].strip()
                        if all(c.isupper() or c.isspace() for c in siguiente_linea) and len(siguiente_linea) > 4:
                            datos["Apellidos"] = siguiente_linea
                            logger.debug(f"Apellidos encontrados después del número: {siguiente_linea}")
                            break
                    break

        # Buscar nombres por palabra clave
        for i, linea in enumerate(lineas):
            if "NOMBRES" in linea.upper():
                for j in range(i + 1, len(lineas)):
                    siguiente_linea = lineas[j].strip()
                    if all(c.isupper() or c.isspace() for c in siguiente_linea) and len(siguiente_linea) > 2:
                        datos["Nombres"] = siguiente_linea
                        logger.debug(f"Nombres encontrados por palabra clave: {siguiente_linea}")
                        break
                break

        # Si no se encontraron nombres, buscar después de apellidos
        if "Nombres" not in datos and "Apellidos" in datos:
            for i, linea in enumerate(lineas):
                if linea.strip() == datos["Apellidos"]:
                    for j in range(i + 1, len(lineas)):
                        siguiente_linea = lineas[j].strip()
                        if all(c.isupper() or c.isspace() for c in siguiente_linea) and len(siguiente_linea) > 2:
                            datos["Nombres"] = siguiente_linea
                            logger.debug(f"Nombres encontrados después de apellidos: {siguiente_linea}")
                            break
                    break

    elif tipo_cedula == "nueva":
        # Buscar NUIP
        for i, linea in enumerate(lineas):
            if "NUIP" in linea.upper():
                if i + 1 < len(lineas):
                    nuip = lineas[i + 1].strip()
                    if es_dato_valido(nuip, "numero"):
                        datos["NUIP"] = nuip
                        logger.debug(f"NUIP encontrado: {nuip}")
                        # Buscar apellidos después del NUIP
                        for j in range(i + 2, len(lineas)):
                            siguiente_linea = lineas[j].strip()
                            if (es_dato_valido(siguiente_linea, "apellidos") and
                                not any(palabra in siguiente_linea.upper() for palabra in palabras_excluidas)):
                                datos["Apellidos"] = siguiente_linea
                                logger.debug(f"Apellidos encontrados después del NUIP: {siguiente_linea}")
                                break
                break
        
        # Buscar nombres
        for i, linea in enumerate(lineas):
            if "NOMBRES" in linea.upper():
                if i + 1 < len(lineas):
                    siguiente_linea = lineas[i + 1].strip()
                    if not any(palabra in siguiente_linea.upper() for palabra in palabras_excluidas):
                        datos["Nombres"] = siguiente_linea
                        logger.debug(f"Nombres encontrados: {siguiente_linea}")
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