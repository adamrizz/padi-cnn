import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import uvicorn
import requests
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ui-padi-cnn.vercel.app/"],  # Ganti sesuai domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL model dari GitHub Releases
MODEL_URL = "https://github.com/adamrizz/padi-cnn/releases/download/v1.0/daun_padi_cnn_model.keras"
MODEL_PATH_LOCAL = "daun_padi_cnn_model.keras"

# Fungsi download model dari GitHub
def download_model_from_github(url, destination):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Gagal mengunduh model dari GitHub: Status code {response.status_code}")
    with open(destination, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    # Validasi file: pastikan bukan file HTML
    if os.path.getsize(destination) < 100000:
        with open(destination, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "<html" in content:
                raise RuntimeError("File hasil unduhan bukan file .keras, tapi halaman HTML. Cek link GitHub Releases-nya.")

# Unduh model jika belum ada
if not os.path.exists(MODEL_PATH_LOCAL):
    print("Mengunduh model dari GitHub Releases...")
    try:
        download_model_from_github(MODEL_URL, MODEL_PATH_LOCAL)
        print("Model berhasil diunduh.")
    except Exception as e:
        raise RuntimeError(f"Gagal mengunduh model: {e}")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH_LOCAL)
    print("Model berhasil dimuat.")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model Keras dari {MODEL_PATH_LOCAL}: {e}")

# Konfigurasi input/output model
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
CLASS_NAMES = [
    "Bacterial Leaf Blight", "Leaf Blast", "Leaf Scald",
    "Brown Spot", "Narrow  Brown Spot", "Healthy"
]

@app.get("/")
async def home():
    return {"message": "API Klasifikasi Daun Padi siap digunakan."}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        print("STEP 1: File diterima:", file.filename)
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File harus berupa gambar.")

        contents = await file.read()
        print("STEP 2: File dibaca, panjang:", len(contents))

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        print("STEP 3: Gambar dikonversi ke RGB")

        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        print("STEP 4: Gambar diresize")

        img_array = np.expand_dims(np.array(image), axis=0) / 255.0
        print("STEP 5: Gambar diproses menjadi array")

        predictions = model.predict(img_array)
        print("STEP 6: Prediksi selesai")

        score = tf.nn.softmax(predictions[0])
        predicted_index = int(np.argmax(score))
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(np.max(score))

        print("STEP 7: Prediksi:", predicted_label, "Confidence:", confidence)

        return {
            "filename": file.filename,
            "predicted_class": predicted_label,
            "confidence": confidence,
            "all_predictions": {
                CLASS_NAMES[i]: float(score[i]) for i in range(len(CLASS_NAMES))
            }
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default ke 8000 jika tidak diset Railway
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
