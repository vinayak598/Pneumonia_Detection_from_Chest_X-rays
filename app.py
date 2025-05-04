import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_cnn_model.h5")
    return model

model = load_model()

st.title("\U0001FA7A Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the model will predict if it indicates **Pneumonia** or **Normal**.")

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

def predict_image(image_data):
    img = image_data.resize((150, 150))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
    result = predict_image(image)
    st.success(f"Prediction: **{result}**")

