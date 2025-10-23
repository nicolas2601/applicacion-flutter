#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 + EasyOCR para detección de placas
# Optimizado para producción en Coolify

import os
import logging
import base64
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

# -------------------------
# Config / Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("yolo-plates")

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))
RETURN_IMAGE = os.getenv("RETURN_IMAGE", "true").lower() == "true"

# -------------------------
# App init
# -------------------------
app = FastAPI(
    title="YOLOv8 - Detector de Placas (OCR)",
    description="API para detección de placas vehiculares usando YOLOv8 y EasyOCR",
    version="1.0.0"
)

# CORS: Permitir todas las origins en desarrollo, restringir en producción
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Cargar modelo y OCR (al inicio)
# -------------------------
logger.info("🔹 Cargando modelo YOLOv8 desde %s ...", MODEL_PATH)
try:
    model = YOLO(MODEL_PATH)
    logger.info("✅ Modelo YOLOv8 cargado correctamente.")
except Exception as e:
    logger.error("❌ Error cargando modelo: %s", e)
    raise

logger.info("🔹 Inicializando EasyOCR con idiomas: %s", OCR_LANGS)
try:
    reader = easyocr.Reader(OCR_LANGS, gpu=False)
    logger.info("✅ EasyOCR listo.")
except Exception as e:
    logger.error("❌ Error inicializando EasyOCR: %s", e)
    raise

# -------------------------
# Helpers
# -------------------------
def ocr_read_text_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    """Ejecuta EasyOCR sobre un ROI y devuelve texto limpio."""
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        result = reader.readtext(roi_rgb)
        if not result:
            return None
        best = max(result, key=lambda x: x[2])
        text = best[1]
        # Limpiar: solo alfanuméricos
        text = "".join(ch for ch in text if ch.isalnum())
        return text.upper() if text else None
    except Exception as e:
        logger.exception("OCR error: %s", e)
        return None


def image_to_base64_jpg(img_bgr: np.ndarray) -> str:
    """Convierte imagen BGR a base64 (JPG)."""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode('utf-8')


# -------------------------
# Rutas
# -------------------------
@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "YOLOv8 + OCR server running",
        "status": "healthy",
        "version": "1.0.0",
        "model": MODEL_PATH
    }


@app.get("/health")
def health():
    """Health check para Docker/Coolify"""
    return {"status": "ok"}


@app.post("/predict/")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Endpoint principal para detección de placas.
    
    Acepta:
    - Multipart file upload (form-data)
    - Base64 string (form field)
    
    Retorna:
    {
        "success": true,
        "placas": ["ABC123", "XYZ987"],
        "num_placas": 2,
        "image": "base64...",
        "message": "OK"
    }
    """
    try:
        logger.info("📩 Petición recibida en /predict/")

        # Leer imagen
        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
        elif image_base64:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]
            image_base64 = image_base64.strip()
            try:
                img_data = base64.b64decode(image_base64 + "===")
            except Exception as e:
                logger.error("❌ Base64 inválido: %s", e)
                return JSONResponse(
                    status_code=400,
                    content={"error": "Base64 inválido o corrupto"}
                )
            nparr = np.frombuffer(img_data, np.uint8)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "No se recibió ninguna imagen"}
            )

        # Decodificar
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No se pudo decodificar la imagen"}
            )

        # Detección YOLO
        logger.info("🧠 Procesando con YOLOv8...")
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)
        
        if not results:
            return {
                "success": True,
                "placas": [],
                "num_placas": 0,
                "image": None,
                "message": "Sin detecciones"
            }

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        clss = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])

        placas_detectadas: List[str] = []
        h, w = frame.shape[:2]

        # Procesar cada detección
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(clss[i]) if len(clss) > i else None
            label = model.names[cls_id] if cls_id is not None else "objeto"
            conf = confs[i] if len(confs) > i else 0

            # Recortar ROI
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            roi = frame[y1c:y2c, x1c:x2c].copy()

            # OCR solo para placas
            if any(k in label.lower() for k in ["placa", "plate", "license"]):
                text_detected = ocr_read_text_from_roi(roi)
                if text_detected:
                    placas_detectadas.append(text_detected)
                    # Dibujar texto en amarillo
                    cv2.putText(
                        frame, text_detected, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
                    )

            # Dibujar caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
            )

        # Codificar imagen resultante
        img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None

        logger.info("✅ Placas detectadas: %s", placas_detectadas)

        return {
            "success": True,
            "placas": placas_detectadas,
            "num_placas": len(placas_detectadas),
            "image": img_b64,
            "message": "OK" if placas_detectadas else "No se detectaron placas"
        }

    except Exception as e:
        logger.exception("Error en /predict/: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error interno: {str(e)}"}
        )


# -------------------------
# Main (para desarrollo local)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("🚀 Iniciando servidor en 0.0.0.0:%s", port)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)