import os
import requests
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np

app = FastAPI()

# URL model dari Google Drive (akan diambil dari environment variable di Railway)
# Jika tidak ada env var, gunakan fallback URL (bisa diisi dengan URL default atau kosongkan)
MODEL_URL = os.environ.get("MODEL_URL", "https://drive.google.com/uc?export=download&id=13WBSxCpDo466a-eNecpd6WB3K-B2KAlC")
MODEL_PATH_LOCAL = 'daun_padi_cnn_model.keras'

# Download model jika belum ada di lokal
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
        print(f"Gagal mengunduh model: {e}. Pastikan URL benar dan file dapat diakses.")
        raise RuntimeError(f"Gagal mengunduh model dari {MODEL_URL}: {e}")

# Muat model Keras dari path lokal yang sudah diunduh
try:
    model = tf.keras.models.load_model(MODEL_PATH_LOCAL)
    print(f"Model berhasil dimuat dari: {MODEL_PATH_LOCAL}")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    raise RuntimeError(f"Gagal memuat model Keras dari {MODEL_PATH_LOCAL}: {e}")

# Definisikan ukuran input yang diharapkan oleh model Anda
IMAGE_HEIGHT = 150 # Contoh, ganti dengan tinggi gambar model Anda
IMAGE_WIDTH = 150  # Contoh, ganti dengan lebar gambar model Anda

# Definisikan kelas (label) output dari model Anda
CLASS_NAMES = ["Bacterial Leaf Blight", "Leaf Blast","Leaf Scald", "Brown Spot","Narrow  Brown Spot", "Healthy"] # Ganti dengan daftar kelas Anda

@app.get("/")
async def read_root():
    return {"message": "Selamat datang di API Klasifikasi Daun Padi!"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.expand_dims(np.array(image), axis=0)
        img_array = img_array / 255.0 # Hapus jika tidak ada normalisasi

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

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