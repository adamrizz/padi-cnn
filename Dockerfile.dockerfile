# Gunakan image Python resmi sebagai base image
FROM python:3.9-slim-buster

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt dan instal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin aplikasi FastAPI Anda dan file model ke dalam container
COPY main.py .

# Expose port yang akan digunakan oleh aplikasi FastAPI (default Uvicorn)
EXPOSE 8000

# Perintah untuk menjalankan aplikasi FastAPI menggunakan Uvicorn
# Gunakan --host 0.0.0.0 agar aplikasi dapat diakses dari luar container
# dan --port $PORT untuk menggunakan port yang disediakan oleh Railway
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]