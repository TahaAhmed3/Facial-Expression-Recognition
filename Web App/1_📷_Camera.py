import streamlit as st
import cv2
import numpy as np
import keras


IMG_SIZE = 224
emotion = { 0: "angry", 1: "happy", 2: "neutral", 3: "sad", 4: "surprise" }

model = keras.models.load_model("FER_Model.h5")

def prepare_image(image):
    image = keras.applications.resnet_v2.preprocess_input(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

def StartVideoCapture():
    cap = cv2.VideoCapture(0)
    while cap.isOpened and not stop_button:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

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
            cv2.putText(frame, emotion[maxindex] + f" [{round(emotion_prediction[0][maxindex] * 100)}%]", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
        frame_placeholder.image(frame, channels="BGR")

    cap.release()



st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🟦",
)

st.title("Video Capture")
col1, col2 = st.columns(2)
start_button = col1.button("Start")
stop_button = col2.button("Stop")
frame_placeholder = st.empty()


if start_button:
    StartVideoCapture()

      





