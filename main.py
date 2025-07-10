from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import requests


MODEL_URL = os.environ.get("MODEL_URL", "https://drive.google.com/file/d/13WBSxCpDo466a-eNecpd6WB3K-B2KAlC/view?usp=sharing") # Ganti dengan URL model Anda
MODEL_PATH_LOCAL = 'daun_padi_cnn_model.keras'

# Download model jika belum ada
if not os.path.exists(MODEL_PATH_LOCAL):
    print(f"Mengunduh model dari {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status() # Pastikan request berhasil
        with open(MODEL_PATH_LOCAL, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model berhasil diunduh.")
    except Exception as e:
        print(f"Gagal mengunduh model: {e}")
        raise RuntimeError(f"Gagal mengunduh model dari {MODEL_URL}: {e}")

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Path ke model Keras Anda. Pastikan nama file ini sesuai dengan model Anda.
MODEL_URL = os.environ.get("MODEL_URL", "https://drive.google.com/file/d/13WBSxCpDo466a-eNecpd6WB3K-B2KAlC/view?usp=sharing") # Ganti dengan URL model Anda
MODEL_PATH = 'daun_padi_cnn_model.keras'

# Muat model Keras
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    # Jika model gagal dimuat, aplikasi tidak dapat berjalan.
    # Anda bisa memilih untuk raise exception atau hanya print error.
    # Untuk deployment, lebih baik aplikasi gagal startup jika model tidak ada.
    raise RuntimeError(f"Gagal memuat model Keras dari {MODEL_PATH}: {e}")

# Definisikan ukuran input yang diharapkan oleh model Anda
# Sesuaikan dengan ukuran gambar yang Anda gunakan saat melatih model
IMAGE_HEIGHT = 150 # Contoh, ganti dengan tinggi gambar model Anda
IMAGE_WIDTH = 150  # Contoh, ganti dengan lebar gambar model Anda

# Definisikan kelas (label) output dari model Anda
# Sesuaikan dengan kelas yang Anda latih (misalnya, Sehat, Bercak Coklat, dll.)
CLASS_NAMES = ["Bacterial Blight", "Blast", "Brown Spot", "Healthy"] # Ganti dengan daftar kelas Anda

# Endpoint Root untuk menguji apakah API berjalan
@app.get("/")
async def read_root():
    return {"message": "Selamat datang di API Klasifikasi Daun Padi!"}

# Endpoint untuk prediksi gambar
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    try:
        # Baca gambar dari request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize gambar ke ukuran yang diharapkan model
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

        # Konversi gambar ke array NumPy dan normalisasi (sesuai pre-processing model Anda)
        img_array = np.array(image)
        # Model seringkali mengharapkan input dengan batch dimension (misal: (1, H, W, C))
        img_array = np.expand_dims(img_array, axis=0)

        # Normalisasi: Jika model Anda dilatih dengan gambar dinormalisasi ke 0-1, lakukan ini:
        img_array = img_array / 255.0 # Hapus baris ini jika model Anda tidak dinormalisasi

        # Lakukan prediksi
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # Ambil softmax dari output pertama jika multi-kelas

        # Dapatkan kelas dengan probabilitas tertinggi
        predicted_class_index = np.argmax(score)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(score))

        return {
            "filename": file.filename,
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "all_predictions": {CLASS_NAMES[i]: float(score[i]) for i in range(len(CLASS_NAMES))}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {e}")