import streamlit as st
import cv2
import numpy as np
import keras
import os


IMG_SIZE = 224
emotion = { 0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprise" }
model = keras.models.load_model("FER_Model.h5")

def prepare_image(image):
    image = keras.applications.resnet_v2.preprocess_input(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

def Predict():
    global filename, processed_video_path

    cap = cv2.VideoCapture(uploaded_file_path)
    filename = "processed_video.mp4"
    processed_video_path = "temp_files/" + filename

    fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height), isColor=True)
    
    ret = False
    if cap.isOpened():
        ret = True
    while ret:
        ret, frame = cap.read()

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
        
        for (x, y, w, h) in faces:
            # roi: region of interest
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = frame[y:y+h, x:x+h]

            
            cropped_img = np.expand_dims(prepare_image(roi), axis=0)
            # predict the emotions
            emotion_prediction = model.predict(cropped_img, verbose=0)
            maxindex = np.argmax(emotion_prediction, axis=1)[0]
            cv2.putText(frame, emotion[maxindex], (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
        out.write(frame)

    cap.release()
    out.release()
    


st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🟦",
)

st.title("Upload Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    uploaded_file_path = "temp_files/" + uploaded_file.name
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = st.text("Processing")
    Predict()
    text.empty()
    
    processed_video_file = open(processed_video_path, "rb")
    st.video(processed_video_file)
    download_btn = st.download_button(
            label="Download",
            data=processed_video_file,
            file_name=filename,
            key="download_button",
        )
    processed_video_file.close()
    os.remove(processed_video_path)
    os.remove(uploaded_file_path)



