import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import requests
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Konfigurasi ID dan lokasi model
FILE_ID = "13WBSxCpDo466a-eNecpd6WB3K-B2KAlC"
MODEL_PATH_LOCAL = "daun_padi_cnn_model.keras"
GOOGLE_DRIVE_DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

# Fungsi untuk ambil token konfirmasi dari Google Drive
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

# Simpan response ke file
def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Fungsi download model dari Google Drive
def download_model_from_drive(file_id, destination):
    session = requests.Session()
    response = session.get(GOOGLE_DRIVE_DOWNLOAD_URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(GOOGLE_DRIVE_DOWNLOAD_URL, params=params, stream=True)

    save_response_content(response, destination)

    # Validasi file
    if os.path.getsize(destination) < 100000:  # < 100KB kemungkinan besar HTML
        with open(destination, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if "<html" in content:
                raise RuntimeError("File yang diunduh bukan file .keras, tapi halaman HTML. Cek apakah file Drive public.")

# Unduh model jika belum ada
if not os.path.exists(MODEL_PATH_LOCAL):
    try:
        print("Mengunduh model dari Google Drive...")
        download_model_from_drive(FILE_ID, MODEL_PATH_LOCAL)
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
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.expand_dims(np.array(image), axis=0) / 255.0

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_index = int(np.argmax(score))
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
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {e}")
