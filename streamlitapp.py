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

# Tempat tetap untuk gambar
placeholder = st.empty()

# Upload gambar
uploaded_file = st.file_uploader("Silahkan Upload Gambar", type=["jpg", "jpeg", "png"])

# Placeholder untuk hasil
hasil_placeholder = st.empty()

if uploaded_file is not None:
    # Buka dan tampilkan gambar dengan ukuran tetap
    image = Image.open(uploaded_file)
    image_resized = image.resize((300, 300))  # Resize gambar menjadi 300x300
    with placeholder.container():
        st.image(image_resized, caption="Uploaded Image", use_column_width=False)

    # Preprocess gambar
    processed_image = preprocess_image(image, target_size=(224, 224))

    # Prediksi
    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Tampilkan hasil dengan gaya
    hasil_placeholder.markdown(
        f"""
        <div style="text-align: center; font-size: 20px;">
            <strong>Hasil:</strong> 
            <span style="font-size: 24px; color: green;"><strong>Prediksi:</strong> {predicted_class}</span> 
            <span style="font-size: 24px; color: blue;"><strong>Confidence:</strong> {confidence:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Placeholder kosong untuk hasil
    hasil_placeholder.markdown(
        """
        <div style="text-align: center; font-size: 20px;">
            <strong>Hasil:</strong> 
            <span style="font-size: 24px; color: gray;"><strong>Prediksi:</strong> -</span> 
            <span style="font-size: 24px; color: gray;"><strong>Confidence:</strong> -</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
