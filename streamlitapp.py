import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import requests
import os

# URL model
MODEL_URL = "https://raw.githubusercontent.com/jeremyfelix12/test-1/main/TA2.h5"

# Fungsi untuk mengunduh model
def download_model(url, save_path):
    """Download the model file from the given URL."""
    response = requests.get(url)
    with open(save_path, "wb") as file:
        file.write(response.content)

# Unduh model jika belum ada
if not os.path.exists("TA2.h5"):
    download_model(MODEL_URL, "TA2.h5")

# Load model
model = load_model("TA2.h5")

# Label kelas
class_labels = ['Batu Ginjal', 'Kista', 'Normal', 'Tumor']

# Preprocessing gambar
def preprocess_image(image, target_size):
    """Preprocess the uploaded image for prediction."""
    image = image.resize(target_size)  # Resize image
    image = image.convert("L")  # Ubah ke grayscale jika model menggunakan 1 channel
    image = img_to_array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan batch dimension
    return image

# Streamlit App
st.title("Klasifikasi Penyakit Ginjal")

# Upload gambar
uploaded_file = st.file_uploader("Silahkan Upload Gambar", type=["jpg", "jpeg", "png"])

# Tampilkan label hasil dari awal
st.markdown("<h3 style='margin-top:20px;'>Hasil:</h3>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    resized_image = image.resize((320, 320))  # Resize gambar ke 320x320
    st.image(resized_image, caption="Uploaded Image", use_column_width=False)

    # Preprocess gambar
    processed_image = preprocess_image(image, target_size=(224, 224))

    # Prediksi
    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Tampilkan hasil dengan format yang diminta
    result_html = f"""
    <div style="display: flex; gap: 20px; align-items: center; margin-top: 20px;">
        <h3 style="margin: 0;">Prediksi:</h3>
        <h3 style="margin: 0; color: #007BFF;">{predicted_class}</h3>
        <h3 style="margin: 0;">Confidence:</h3>
        <h3 style="margin: 0; color: #28A745;">{confidence:.2f}</h3>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
