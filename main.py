import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import requests
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Konfigurasi
FILE_ID = "13WBSxCpDo466a-eNecpd6WB3K-B2KAlC"
MODEL_PATH_LOCAL = "daun_padi_cnn_model.keras"
DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

# Fungsi untuk mendapatkan token konfirmasi Google Drive
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# Fungsi menyimpan file dari response
def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Unduh model jika belum ada
def download_model():
    session = requests.Session()
    response = session.get(DOWNLOAD_URL, params={'id': FILE_ID}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': FILE_ID, 'confirm': token}
        response = session.get(DOWNLOAD_URL, params=params, stream=True)

    save_response_content(response, MODEL_PATH_LOCAL)

    # Cek apakah file benar-benar file model, bukan HTML error
    if os.path.getsize(MODEL_PATH_LOCAL) < 100000:  # <100KB kemungkinan besar HTML
        with open(MODEL_PATH_LOCAL, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "<html" in content:
                raise RuntimeError("Gagal mengunduh model: file bukan model yang valid, tapi halaman HTML. Cek permission Google Drive.")

# Unduh dan load model
if not os.path.exists(MODEL_PATH_LOCAL):
    try:
        print("Mengunduh model dari Google Drive...")
        download_model()
        print("Model berhasil diunduh.")
    except Exception as e:
        raise RuntimeError(f"Gagal mengunduh model: {e}")

try:
    model = tf.keras.models.load_model(MODEL_PATH_LOCAL)
    print("Model berhasil dimuat.")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model Keras dari {MODEL_PATH_LOCAL}: {e}")

# Setup model config
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
CLASS_NAMES = [
    "Bacterial Leaf Blight", "Leaf Blast", "Leaf Scald",
    "Brown Spot", "Narrow  Brown Spot", "Healthy"
]

@app.get("/")
async def read_root():
    return {"message": "API Klasifikasi Daun Padi siap digunakan."}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.expand_dims(np.array(image), axis=0) / 255.0

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_index = np.argmax(score)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(np.max(score))

        return {
            "filename": file.filename,
            "predicted_class": predicted_label,
            "confidence": confidence,
            "all_predictions": {
                CLASS_NAMES[i]: float(score[i]) for i in range(len(CLASS_NAMES))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {e}")
