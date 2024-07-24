import cv2 
import streamlit as st
import numpy as np
from PIL import Image


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def draw_rectangle(image, faces):
     for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x + w, y + h), (0, 0, 255), 2)


def detect_faces_in_image(upl_image, scale_factor, min_neighbors, min_size):
    image_array = np.array(Image.open(upl_image))
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
            gray_image, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size
        )    
    draw_rectangle(image_array, faces)

    st.image(image_array, channels="BGR", use_column_width=True)


def detect_faces_from_cam(scale_factor, min_neighbors, min_size):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Cam", key="stop")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
                    gray_image, 
                    scaleFactor=scale_factor, 
                    minNeighbors=min_neighbors, 
                    minSize=min_size
                )
        draw_rectangle(frame, faces)

        stframe.image(frame, channels="BGR", use_column_width=True)
        if stop_button:
            break

    cap.release()


st.title("Face Detection Example")
st.subheader("Open Camera or Upload Photo")

scale_factor = st.sidebar.slider("Scale Factor", 1.1, 2.0, 1.3)
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 20, 10)
min_size = st.sidebar.slider("Min Size", 10, 100, 30, step=10)

if st.button("Open Cam"):
    detect_faces_from_cam(scale_factor, min_neighbors, (min_size, min_size))

uploaded = st.file_uploader("Upload", ["jpg", "jpeg", "png"])

if uploaded is not None:
    detect_faces_in_image(uploaded, scale_factor, min_neighbors, (min_size, min_size))
