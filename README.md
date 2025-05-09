# 🦪 Pneumonia Detection Using CNN & Chest X-rays

This project uses a Convolutional Neural Network (CNN) to detect Pneumonia from chest X-ray images using the [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## 🔧 Features

- Automatic dataset download using `kagglehub`
- CNN model built using TensorFlow/Keras
- Streamlit web app to upload images and get predictions

## 💪 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

if not working

```bash
pip install streamlit tensorflow numpy pillow kagglehub
```

### 2. Train the model (optional)

```bash
python pneumonia_cnn.py
```

### 3. Launch Streamlit app

```bash
streamlit run app.py
```

## 📅 File Structure

- `pneumonia_cnn.py`: Model training & dataset download
- `app.py`: Streamlit app for image upload & prediction
- `requirements.txt`: Dependencies
- `pneumonia_cnn_model.h5`: Saved trained model

## ✅ Example

Upload a chest X-ray image and receive a prediction: `NORMAL` or `PNEUMONIA`.

---

Built with ❤️ using TensorFlow and Streamlit.
#   P n e u m o n i a _ D e t e c t i o n _ f r o m _ C h e s t _ X - r a y s  
 