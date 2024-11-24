import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah dengan ukuran 224x224 pixel
    image = Image.open(uploaded_file)
    resized_image = image.resize((224, 224))
    st.image(resized_image, caption="Uploaded Image (Resized to 224x224)", use_column_width=False)

    # Preprocess gambar
    processed_image = preprocess_image(image, target_size=(224, 224))

    # Prediksi
    predictions = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Placeholder untuk hasil prediksi
    predicted_class_placeholder = st.markdown(
        "<h3>Predicted Class: <span style='color: gray;'>Belum tersedia</span></h3>", unsafe_allow_html=True
    )
    confidence_placeholder = st.markdown(
        "<h3>Confidence: <span style='color: gray;'>Belum tersedia</span></h3>", unsafe_allow_html=True
    )
    
    # Perbarui hasil prediksi dan confidence
    predicted_class_placeholder.markdown(
        f"<h3>Predicted Class: <span style='color: blue;'>{predicted_class}</span></h3>", unsafe_allow_html=True
    )
    confidence_placeholder.markdown(
        f"<h3>Confidence: <span style='color: blue;'>{confidence:.2f}</span></h3>", unsafe_allow_html=True
    )
