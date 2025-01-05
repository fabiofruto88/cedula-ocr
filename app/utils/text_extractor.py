import re
import pytesseract

class TextExtractor:
    @staticmethod
    def extract_info(text):
        """Extraer información relevante del texto usando expresiones regulares"""
        info = {
            "nombre": None,
            "numero_cedula": None,
            "fecha_nacimiento": None,
        }
        
        # Buscar número de cédula
        cedula_match = re.search(r'\b\d{8,10}\b', text)
        if cedula_match:
            info["numero_cedula"] = cedula_match.group()
        
        # Buscar fecha
        fecha_match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', text)
        if fecha_match:
            info["fecha_nacimiento"] = fecha_match.group()
        
        # Extraer nombre
        lines = [line.strip() for line in text.split('\n')]
        nombre_candidates = [
            line for line in lines 
            if len(line) > 10 and not any(c.isdigit() for c in line)
        ]
        if nombre_candidates:
            info["nombre"] = max(nombre_candidates, key=len)
        
        return info

    @staticmethod
    def process_image(image):
        """Procesar imagen con Tesseract"""
        text = pytesseract.image_to_string(image, lang='spa')
        return text