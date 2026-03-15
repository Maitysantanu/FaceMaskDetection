import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Mask Detection",
    page_icon="😷",
    layout="centered"
)


# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_my_model():
    model = load_model("model.keras", compile=False)
    return model


with st.spinner("Loading AI model..."):
    model = load_my_model()


# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess(img):

    img = cv2.resize(img, (128,128))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# ----------------------------
# Prediction
# ----------------------------
def predict(img):

    img = preprocess(img)

    pred = model.predict(img)

    pred_label = np.argmax(pred)

    confidence = np.max(pred)

    return pred_label, confidence


# ----------------------------
# UI
# ----------------------------
st.title("😷 Face Mask Detection")

st.write("Upload an image or capture from camera to detect mask.")


option = st.radio(
    "Choose Input Method",
    ("Upload Image", "Use Camera")
)


# ----------------------------
# Upload Image
# ----------------------------
if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        img = np.array(image)

        st.image(image, caption="Uploaded Image", width=250)

        with st.spinner("Analyzing image..."):

            pred_label, confidence = predict(img)

        st.write("Confidence:", float(confidence))

        if pred_label == 1:

            st.success("The person is wearing mask 😷")

        else:

            st.error("The person is not wearing mask ❌")


# ----------------------------
# Camera Input
# ----------------------------
if option == "Use Camera":

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")

        img = np.array(image)

        st.image(image, caption="Captured Image", use_column_width=True)

        with st.spinner("Analyzing image..."):

            pred_label, confidence = predict(img)

        st.write("Confidence:", float(confidence))

        if pred_label == 1:

            st.success("The person is wearing mask 😷")

        else:

            st.error("The person is not wearing mask ❌")