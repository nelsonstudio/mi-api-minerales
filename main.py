from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request

app = FastAPI(title="API Minerales IA")

# =========================
# 🔥 CONFIGURACIÓN
# =========================
MODEL_PATH = "modelo_minerales.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1FeMP_w-0ci5YCvE-9EFPniJTy8kfWfJ5"

CLASES = ["galena", "pirita", "calcopirita"]

# =========================
# 📥 DESCARGAR MODELO
# =========================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo desde Drive...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado")

# =========================
# 🧠 CARGAR MODELO (1 sola vez)
# =========================
model = None

def load_model():
    global model
    if model is None:
        download_model()
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
        return {"error": str(e)}

# =========================
# 🚀 INICIO (Render)
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
