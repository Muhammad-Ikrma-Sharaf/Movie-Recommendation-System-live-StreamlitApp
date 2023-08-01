import cv2
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
# Load the pre-trained YOLO model
# Create an instance of the YOLO model
model = YOLO("last.pt")

# Set Streamlit app title and description
st.title("Brain Tumor Segmentation")
st.write("Upload an image and the model will perform object detection.")

# Allow user to upload an image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    #st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection on the uploaded image using the YOLO model
    output_image = model.predict(image)
    res_plotted = output_image[0].plot()
    image = Image.fromarray(res_plotted)
    # Display the output image with detected objects
    st.image(image, caption="Output Image", use_column_width=True)