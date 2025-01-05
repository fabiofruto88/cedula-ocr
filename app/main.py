from fastapi import FastAPI, File, UploadFile
from .services.image_processor import ImageProcessor
from .utils.text_extractor import TextExtractor

app = FastAPI(title="OCR Cédula API")

@app.post("/extract_cedula")
async def extract_cedula(file: UploadFile = File(...)):
    try:
        # Leer la imagen
        contents = await file.read()
        
        # Preprocesar la imagen
        processed_img = ImageProcessor.preprocess_image(contents)
        
        # Extraer texto
        text = TextExtractor.process_image(processed_img)
        
        # Extraer información relevante
        info = TextExtractor.extract_info(text)
        
        return {
            "success": True,
            "data": info,
            "raw_text": text
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)