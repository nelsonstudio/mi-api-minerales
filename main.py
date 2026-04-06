from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = FastAPI(title="API Minerales IA")

# =========================
# 🔥 CONFIGURACIÓN
# =========================
MODEL_PATH = "modelo_minerales.h5"

# Cambia según tus clases
CLASES = ["galena", "pirita", "calcopirita"]

# =========================
# 🧠 CARGAR MODELO (solo 1 vez)
# =========================
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

# =========================
# 🖼️ PREPROCESAMIENTO
# =========================
def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# 🏠 RUTA PRINCIPAL
# =========================
@app.get("/")
def home():
    return {"mensaje": "API activa - Minerales IA"}

# =========================
# 🔍 PREDICCIÓN
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        img = preprocess(image)

        modelo = load_model()
        pred = modelo.predict(img)

        clase_idx = int(np.argmax(pred))
        confianza = float(np.max(pred))

        return {
            "mineral": CLASES[clase_idx],
            "confianza": round(confianza, 4)
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# =========================
# 🚀 INICIO (Render compatible)
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
